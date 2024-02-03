(ns scicloj.ml.tribuo-test
  (:require [scicloj.ml.tribuo]
            [scicloj.metamorph.ml :as ml]
            [tech.v3.dataset.modelling :as dsmod]
            [scicloj.metamorph.ml.toydata :as data]
            [scicloj.metamorph.core :as mm]

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






(comment

  (require '[scicloj.ml.tribuo]
           '[scicloj.metamorph.ml :as ml]
           '[tech.v3.dataset.modelling :as dsmod]
           '[scicloj.metamorph.ml.toydata :as data]
           '[scicloj.metamorph.core :as mm])


  ;;
  (def iris (data/iris-ds))
  (def split (dsmod/train-test-split iris))
  (def model (ml/train (:train-ds split)
                       {:model-type :scicloj.ml.tribuo/classification
                        :tribuo-components [{:name "trainer"
                                             :type "org.tribuo.classification.dtree.CARTClassificationTrainer"}]
                        :tribuo-trainer-name "trainer"}))
  (def prediction (ml/predict (:test-ds split) model))




  ;;  the same using metamorh pipelies

  (def cart-pipeline
    (mm/pipeline
     (ml/model {:model-type :scicloj.ml.tribuo/classification
                :tribuo-components [{:name "trainer"
                                     :type "org.tribuo.classification.dtree.CARTClassificationTrainer"}]
                :tribuo-trainer-name "trainer"})))


  ;;  no global variable needed, as state is in context
  (->> (mm/fit-pipe (:train-ds split) cart-pipeline)
       (mm/transform-pipe (:test-ds split) cart-pipeline)
       :metamorph/data)

  :ok)
