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

            [clojure.test :as t]))

(def simple-iris
  (-> (data/iris-ds)
      (dscat/reverse-map-categorical-xforms)
      (ds/update-column :species (fn [col] (map str col)))
      (dsmod/set-inference-target :species)))



(t/deftest train
  (let [iris simple-iris
        split (dsmod/train-test-split iris {:seed 123})

        model (ml/train (:train-ds split)
                        {:model-type :scicloj.ml.tribuo/classification
                         :tribuo-components [{:name "trainer"
                                              :type "org.tribuo.classification.dtree.CARTClassificationTrainer"}]
                         :tribuo-trainer-name "trainer"})
        predictions (->  (ml/predict (:test-ds split) model))]
                         

    (t/is (= "1"
             (->
              predictions
              (ds-cf/prediction)
              ds/columns
              first
              first)))))


(t/deftest sonar-evaluate
  (let [ds
        (-> (data/sonar-ds))

        make-pipefn (fn  [opts]
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
    (t/is (= 0.7804878048780488
             (->> evaluations
                  flatten
                  (map #(-> % :test-transform :metric))
                  first)))))
