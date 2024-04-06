(ns scicloj.ml.tribuo-test
  (:require [scicloj.ml.tribuo]
            [scicloj.metamorph.ml :as ml]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as dsmod]
            [tech.v3.dataset.categorical :as dscat]
            [scicloj.metamorph.ml.toydata :as data]
            [tech.v3.dataset.column-filters :as ds-cf]
            [scicloj.metamorph.core :as mm]

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
