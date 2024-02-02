(ns scicloj.ml.tribuo-test
  (:require [scicloj.ml.tribuo :as sut]
            [scicloj.metamorph.ml :as ml]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as dsmod]
            [scicloj.metamorph.ml.toydata :as data]
            [clojure.test :as t]))


(t/deftest train
  (let [iris (data/iris-ds)
        split (dsmod/train-test-split iris {:seed 123})

        model (ml/train (:train-ds split)
                        {:model-type :scicloj.ml.tribuo/classification
                         :tribuo-components [{:name "trainer"
                                              :type "org.tribuo.classification.dtree.CARTClassificationTrainer"}]
                         :tribuo-trainer-name "trainer"})]

    (t/is (= "1"
             (->
              (ml/predict (:test-ds split) model)
              :prediction
              seq
              first)))))
