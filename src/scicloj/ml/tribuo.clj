
(ns scicloj.ml.tribuo
  (:require
   [scicloj.metamorph.ml :as ml]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.datatype :as dt]
   [tech.v3.datatype.functional :as fun]
   [fastmath.stats :as stats]
   [tech.v3.datatype.errors :as errors]
   [tech.v3.libs.tribuo :as tribuo])
  (:import [org.tribuo.regression.evaluation RegressionEvaluator]
          [org.tribuo.regression Regressor]))

(defn- make-trainer [options]
  (tribuo/trainer (:tribuo-components options)
                  (:tribuo-trainer-name options)))


(defn- post-process-prediction [ds target-column-name]
  (-> ds
      (ds/assoc-metadata [:prediction] :column-type :prediction)
      (ds/rename-columns {:prediction target-column-name})))




(ml/define-model! :scicloj.ml.tribuo/classification
  (fn [feature-ds target-ds options]

    (let [first-target-col (-> target-ds ds/columns first)
          col-data-type (-> first-target-col meta :datatype)]

      (errors/when-not-errorf
       (or (some? (:categorical-map (meta first-target-col)))
           (= :string col-data-type))

       "Can only handle categorical or :string target column. Target column meta: %s"
       (meta first-target-col)))
    (tribuo/train-classification (make-trainer options) (ds/append-columns feature-ds (ds/columns target-ds))))
    

  (fn [feature-ds thawed-model {:keys [model-data target-columns] :as model}]
    (let [target-column-name (first target-columns)
          prediction
          (->
           (tribuo/predict-classification model-data
                                          feature-ds)
           (post-process-prediction target-column-name))]
      prediction))
  {})

(defn evaluate [model]
  (let [
        feature-ds (-> model :model-data :feature-ds)
        target-ds (-> model :model-data :target-ds)

        model-object (-> model :model-data :model)
        evaluator (RegressionEvaluator.)
        evaluation
        (.evaluate evaluator model-object (tribuo/make-regression-datasource
                                           (ds/append-columns feature-ds (ds/columns target-ds))))]
    evaluation))

(defn ->predictions [evaluation]

  (flatten
   (for [prediction (.getPredictions evaluation)]
     (for [output (iterator-seq (.iterator (.getOutput prediction)))]
       (for [tupple (iterator-seq (.iterator output))]
         {:regressor (.getName tupple)
          :prediction (.getValue tupple)})))))



(defn glance-fn-regression [model]

  (let [

        target-ds (-> model :model-data :target-ds)
        target-column-names (ds-mod/inference-target-column-names target-ds)
        _ (assert (< (count  target-column-names) 2) "Can only handle single inference target")

        evaluation (evaluate model)
        predictions (->predictions evaluation)


        regressor-values (-> target-ds :disease-progression)
        prediction-values (map :prediction predictions)

        target-column-names (ds-mod/inference-target-column-names target-ds)
        regressor-name (name (first target-column-names))
        regressor (Regressor. regressor-name, Double/NaN)]
        


    (ds/->dataset
     {:regressor [ regressor-name]
      :r.squared [(.r2 evaluation regressor)]
      :mae [(.mae evaluation regressor)]
      :rmse [(.rmse evaluation regressor)]
      :rss (stats/rss regressor-values prediction-values)})))


(defn augment-fn-regression [model data]

  (let [prediction-values
        (map
         :prediction
         (->predictions (evaluate model)))

        residuos
        (fun/-
         (get data
              (first (model :target-columns)))
         prediction-values)]
    (-> data
        (ds/add-column (ds/new-column :.fitted prediction-values))
        (ds/add-column (ds/new-column :.resid residuos)))))


(ml/define-model! :scicloj.ml.tribuo/regression
  (fn [feature-ds target-ds options]
    {:target-ds target-ds
     :feature-ds feature-ds
     :model
     (tribuo/train-regression (make-trainer options) (ds/append-columns feature-ds (ds/columns target-ds)))})

  (fn [feature-ds thawed-model {:keys [model-data target-columns]}]
    (let [model (:model model-data)]
      (->
       (tribuo/predict-regression model feature-ds)
       (post-process-prediction (first target-columns)))))
  {:glance-fn glance-fn-regression
   :augment-fn augment-fn-regression})
