(ns scicloj.ml.tribuo-test
  (:require [scicloj.ml.tribuo]
            [scicloj.metamorph.ml :as ml]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as dsmod]
            [tech.v3.dataset.categorical :as dscat]
            [scicloj.metamorph.ml.toydata :as data]
            [tech.v3.dataset.column-filters :as ds-cf]
            [scicloj.metamorph.core :as mm]
            [scicloj.metamorph.ml.loss :as loss]
            [tablecloth.api :as tc]
            [clojure.test :as t]
            [tech.v3.dataset.categorical :as ds-cat]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.categorical :as categorical]
            [tech.v3.datatype :as dt]))


(def iris-target-raw
  (->>
   (:species (data/iris-ds))
   (map #(case %
           0 :setosa
           1 :versicolor
           2 :virginica))))



(defn make-species-column [datatype categorical?
                           inference-target?
                           species-key->val-map]
  (let [
        meta
        {:categorical? categorical?
         :name :species
         :datatype datatype
         :n-elems 150
         :inference-target? inference-target?
         }]
    (ds/new-column :species (map species-key->val-map iris-target-raw) meta)))


(defn make-iris-ds [species-column result-datatype]
  
  (->
   (assoc (data/iris-ds) :species species-column)
   ((fn [ds]
      (if (some? result-datatype)
        (ds/categorical->number ds [:species] [] result-datatype)
        ds)))
   
   (ds-mod/set-inference-target :species)))

(defn- validate [ds expected-target-val expected-accuracy]
  (let [split (dsmod/train-test-split ds {:seed 123})

        model (ml/train (:train-ds split)
                        {:model-type :scicloj.ml.tribuo/classification
                         :tribuo-components [{:name "trainer"
                                              :type "org.tribuo.classification.dtree.CARTClassificationTrainer"}]
                         :tribuo-trainer-name "trainer"})
        predictions (->  (ml/predict (:test-ds split) model))

        accuracy (loss/classification-accuracy (-> split :test-ds
                                                   dscat/reverse-map-categorical-xforms
                                                   :species)
                                               (-> predictions
                                                   dscat/reverse-map-categorical-xforms
                                                   :species))]


    (t/is (< expected-accuracy accuracy))
    (t/is (= expected-target-val
             (->
              predictions
              (ds-cf/prediction)
              ds/columns
              first
              first)))))

(t/deftest validate-col-variants

  (validate
   (make-iris-ds
    (make-species-column :int32
                         true ;categorical?
                         true ;inference-target? 
                         {:setosa 0
                          :versicolor 1
                          :virginica 2})
    nil)
   1
   0.94)
  (validate
   (make-iris-ds
    (make-species-column :float32
                         true ;categorical?
                         true ;inference-target? 
                         {:setosa 0.0
                          :versicolor 1.0
                          :virginica 2.0})
    nil)
   1.0
   0.94)

  (validate

   (make-iris-ds
    (make-species-column :int64
                         true ;categorical?
                         true ;inference-target? 
                         {:setosa :setosa
                          :versicolor :versicolor
                          :virginica :virginica})
    :float64)
   0.0
   0.94)

  (validate
   (make-iris-ds
    (make-species-column :string
                         true ;categorical?
                         true ;inference-target? 
                         {:setosa "0"
                          :versicolor "1"
                          :virginica "2"})
    :float64)
   1.0
   0.94)

  (validate
   (make-iris-ds
    (make-species-column :string
                         true ;categorical?
                         true ;inference-target? 
                         {:setosa "0"
                          :versicolor "1"
                          :virginica "2"})
    nil)
   "1"
   0.94)

  (validate
   (make-iris-ds
    (make-species-column :string
                         true ;categorical?
                         true ;inference-target? 
                         {:setosa "77"
                          :versicolor "12"
                          :virginica "2"})
    nil)
   "12"
   0.94)




  (validate
   (make-iris-ds
    (make-species-column :string
                         true ;categorical?
                         true ;inference-target? 
                         {:setosa :setosa
                          :versicolor :versicolor
                          :virginica :virginica})
    :int32)
   0
   0.94)

  (validate
   (make-iris-ds
    (make-species-column :string
                         true ;categorical?
                         true ;inference-target? 
                         {:setosa :setosa
                          :versicolor :versicolor
                          :virginica :virginica})
    :float32)
   0.0
   0.94)

  (validate
   (make-iris-ds
    (make-species-column :string
                         true ;categorical?
                         true ;inference-target? 
                         {:setosa :setosa
                          :versicolor :versicolor
                          :virginica :virginica})
    nil)
   :versicolor
   0.94)

  (validate
   (make-iris-ds
    (make-species-column :string
                         true ;categorical?
                         true ;inference-target? 
                         {:setosa "setosa"
                          :versicolor "versicolor"
                          :virginica "virginica"})
    nil)
   "versicolor"
   0.94)

  (validate
   (make-iris-ds
    (make-species-column :string
                         true ;categorical?
                         true ;inference-target? 
                         {:setosa false
                          :versicolor true
                          :virginica true})
    nil)
   true
   0.65))


 (t/deftest not-supported 
   (t/is (thrown? Exception
        (validate
         (make-iris-ds
          (make-species-column :string
                               true ;categorical?
                               true ;inference-target? 
                               {:setosa 1
                                :versicolor "12"
                                :virginica "a2a"})
          nil)
         "12"
         0.94)))
   )

 (ds/new-column :x [:a :b :c])

(defn- verify-evaluate [ds]
  (let [make-pipefn (fn  [opts]
                      (mm/pipeline
                       {:metamorph/id  :model}
                       (ml/model {:model-type :scicloj.ml.tribuo/classification
                                  :tribuo-components [{:name "trainer"
                                                       :type "org.tribuo.classification.dtree.CARTClassificationTrainer"
                                                       :properties {:maxDepth "8"}}]



                                  :tribuo-trainer-name "trainer"})))

        splits
        (tc/split->seq ds :kfold {:seed 1234})

        pipefns [(make-pipefn {})]

        evaluations
        (ml/evaluate-pipelines pipefns splits
                               (fn [lhs rhs]
                                 (loss/classification-accuracy lhs rhs))
                               :accuracy
                               {:return-best-crossvalidation-only true
                                :return-best-pipeline-only true
                                :evaluation-handler-fn
                                (fn [eval-result]
                                  eval-result)})]
    (t/is (= 0.8048780487804879
             (->> evaluations
                  flatten
                  (map #(-> % :test-transform :metric))
                  first)))))



(t/deftest sonar-evaluate-2
  (verify-evaluate (-> (data/sonar-ds)
                       (ds/categorical->number [:material]))))
