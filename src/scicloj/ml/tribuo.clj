
(ns scicloj.ml.tribuo
  (:require
   [scicloj.metamorph.ml :as ml]
   [tech.v3.dataset :as ds]
   [tech.v3.datatype.errors :as errors]

   [tech.v3.dataset.column-filters :as ds-cf]
   [tech.v3.dataset.modelling :as ds-mod]

   [tech.v3.libs.tribuo :as tribuo]))

(defn- make-trainer [options]
  (tribuo/trainer (:tribuo-components options)
                  (:tribuo-trainer-name options)))


(defn- post-process-prediction [ds target-column-name]
  (-> ds
      (ds/assoc-metadata [:prediction] :column-type :prediction)
      (ds/rename-columns {:prediction target-column-name})))




(ml/define-model! :scicloj.ml.tribuo/classification
  (fn [feature-ds target-ds options]


    (errors/when-not-errorf (= :string (-> target-ds ds/columns first meta :datatype)) "only target column type :string is supported, target has type: %s" (-> target-ds ds/columns first meta :datatype))
    (tribuo/train-classification (make-trainer options) (ds/append-columns feature-ds (ds/columns target-ds))))



  (fn [feature-ds thawed-model {:keys [model-data target-columns] :as model}]
    (let [target-column-name (first target-columns)
          prediction
          (->
            (tribuo/predict-classification model-data feature-ds)
           (post-process-prediction target-column-name))]
      prediction))
  {})



(ml/define-model! :scicloj.ml.tribuo/regression
  (fn [feature-ds target-ds options]
    (tribuo/train-regression (make-trainer options) (ds/append-columns feature-ds (ds/columns target-ds))))

  (fn [feature-ds thawed-model {:keys [model-data target-columns]}]
    (->
     (tribuo/predict-regression model-data feature-ds)
     (post-process-prediction (first target-columns))))
  {})
