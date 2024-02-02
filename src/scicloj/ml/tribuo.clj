
(ns scicloj.ml.tribuo
  (:require
   [scicloj.metamorph.ml :as ml]
   [tech.v3.dataset :as ds]
   [tech.v3.libs.tribuo :as tribuo]))

(defn- make-trainer [options]
  (tribuo/tribuo-trainer (:tribuo-components options)
                         (:tribuo-trainer-name options)))


(ml/define-model! :scicloj.ml.tribuo/classification
  (fn [feature-ds target-ds options]
    (tribuo/train-classification (make-trainer options) (ds/append-columns feature-ds (ds/columns target-ds))))



  (fn [feature-ds thawed-model {:keys [model-data]}]
    (tribuo/predict-classification model-data feature-ds))
  {})


(ml/define-model! :scicloj.ml.tribuo/regression
  (fn [feature-ds target-ds options]
    (tribuo/train-regression (make-trainer options) (ds/append-columns feature-ds (ds/columns target-ds))))

  (fn [feature-ds thawed-model {:keys [model-data]}]
    (tribuo/predict-regression model-data feature-ds))
  {})
