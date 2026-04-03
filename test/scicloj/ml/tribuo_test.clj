(ns scicloj.ml.tribuo-test
  (:require
   [clojure.test :as t]
   [scicloj.metamorph.core :as mm]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.loss :as loss]
   [scicloj.metamorph.ml.toydata :as data]
   [scicloj.ml.tribuo]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.categorical :as dscat]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.dataset.column-filters :as cf]
   [tech.v3.libs.tribuo :as tribuo]))


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
  (let [meta
        {:categorical? categorical?
         :name :species
         :datatype datatype
         :n-elems 150
         :inference-target? inference-target?}]
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
  (let [options {:model-type :scicloj.ml.tribuo/classification
                 :tribuo-components [{:name "trainer"
                                      :type "org.tribuo.classification.dtree.CARTClassificationTrainer"}]
                 :tribuo-trainer-name "trainer"}

        split (ds-mod/train-test-split ds {:seed 123})

        model (ml/train (:train-ds split) options)
        predictions (->  (ml/predict (:test-ds split) model))

        standardise-fn (:pre-metric-standarisation-fn (ml/options->model-def options))



        accuracy (loss/classification-accuracy (-> split :test-ds
                                                   dscat/reverse-map-categorical-xforms
                                                   :species)
                                               (-> predictions
                                                   dscat/reverse-map-categorical-xforms
                                                   :species))
        score-fn (:score-fn (ml/options->model-def options))
        score (score-fn model (:test-ds split))
        standardised (standardise-fn predictions (:test-ds split) :discrete)
        accuracy-from-standardised (loss/classification-accuracy
                                    (-> standardised :prediction)
                                    (-> standardised :trueth))]

    (t/is (< expected-accuracy accuracy-from-standardised))
    (t/is (< expected-accuracy score))
    (t/is (< expected-accuracy accuracy))
    (t/is (= expected-target-val
             (->
              predictions
              (cf/prediction)
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
   2
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
   2.0
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
   2.0
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
   2.0
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
   2.0
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
   "2"
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
   "2"
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
   2
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
   2.0
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
   :virginica
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
   "virginica"
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
                  0.94))))


(defn- verify-evaluate [ds]
  (let [options
        {:model-type :scicloj.ml.tribuo/classification
         :tribuo-components [{:name "trainer"
                              :type "org.tribuo.classification.dtree.CARTClassificationTrainer"
                              :properties {:maxDepth "8"}}]



         :tribuo-trainer-name "trainer"}
        make-pipefn (fn []
                      (mm/pipeline
                       {:metamorph/id  :model}
                       (ml/model options)))

        splits
        (tc/split->seq ds :kfold {:seed 1234})

        pipefns [(make-pipefn)]

        evaluations
        (ml/evaluate-pipelines pipefns splits
                               (fn [lhs rhs]
                                 (loss/classification-accuracy lhs rhs))
                               :accuracy
                               {:return-best-crossvalidation-only true
                                :return-best-pipeline-only true
                                :evaluation-handler-fn
                                (fn [eval-result]
                                  eval-result)})
        ]

    (t/is (= "org.tribuo.common.tree.TreeModel"
             (->
              evaluations flatten first  :fit-ctx  :model :model-data :model-as-bytes
              (java.io.ByteArrayInputStream.)
              (org.tribuo.Model/deserializeFromStream)
              class
              .getName)))

    (t/is (= "org.tribuo.common.tree.TreeModel"
             (->
              evaluations flatten first  :fit-ctx  :model
              (ml/thaw-model)
              class
              .getName)))

    (t/is (= 0.8048780487804879
             (->> evaluations
                  flatten
                  (map #(-> % :test-transform :metric))
                  first)))))



(t/deftest sonar-evaluate-2
  (verify-evaluate (-> (data/sonar-ds)
                       (ds/categorical->number [:material]))))


(defn- validate-target-symetry [datatype]
  (t/is (= datatype
           (->>
            (ml/train
             (-> (ds/->dataset {:x [1 2 3 4]
                                :y [:a :b :c :d]})
                 (ds/categorical->number [:y] [] datatype)
                 (ds-mod/set-inference-target [:y]))
             {:model-type :scicloj.ml.tribuo/classification
              :tribuo-components [{:name "trainer"
                                   :type "org.tribuo.classification.dtree.CARTClassificationTrainer"
                                   :properties {:maxDepth "8"}}]
              :tribuo-trainer-name "trainer"})
            (ml/predict
             (-> (ds/->dataset {:x [1 2 3 4]})))
            :y
            meta
            :datatype))))


(t/deftest validate-target-sym
  (validate-target-symetry :int8)
  (validate-target-symetry :int16)
  (validate-target-symetry :int32)
  (validate-target-symetry :int64)
  (validate-target-symetry :float32)
  (validate-target-symetry :float64))


(t/deftest xxx
  (let [iris
        (make-iris-ds
         (make-species-column :int32
                              true ;categorical?
                              true ;inference-target? 
                              {:setosa :setosa
                               :versicolor :versicolor
                               :virginica :virginica})
         :int)
        split (ds-mod/train-test-split iris {:seed 123})
        options {:model-type :scicloj.ml.tribuo/classification
                 :tribuo-components [{:name "trainer"
                                      :type "org.tribuo.classification.dtree.CARTClassificationTrainer"}]
                 :tribuo-trainer-name "trainer"}
        model (ml/train (:train-ds split)
                        options)

        p (ml/predict (:test-ds split) model)]
    (t/is (= (-> p cf/prediction ds/column-names) [:species]))
    (t/is (= (-> p cf/probability-distribution ds/column-names) ["virginica" "setosa" "versicolor"]))

    (t/is (.equals
           {:name :species,
            :datatype :int32,
            :n-elems 45,
            :column-type :prediction,
            :categorical-map {:lookup-table {:versicolor 0, :setosa 1, :virginica 2}, :src-column :species, :result-datatype :int}}
           (-> p (cf/prediction) :species meta)))))



