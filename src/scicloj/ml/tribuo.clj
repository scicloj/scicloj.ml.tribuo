(ns scicloj.ml.tribuo
  (:require
   [fastmath.stats :as stats]
   [scicloj.metamorph.ml.loss :as loss]
   [scicloj.metamorph.ml :as ml]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.column :as ds-col]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.datatype :as dt]
   [tech.v3.datatype.errors :as errors]
   [tech.v3.datatype.functional :as fun]
   [tech.v3.libs.tribuo :as tribuo]
   [tech.v3.dataset.categorical :as dscat]
   [tech.v3.dataset.column-filters :as cf])
  (:import
   [org.tribuo.regression Regressor]
   [org.tribuo.regression.evaluation RegressionEvaluator]))

(defn- make-trainer [options]
  (tribuo/trainer (:tribuo-components options)
                  (:tribuo-trainer-name options)))

(defn- string->numeric [element--str column target-datatype target-categorical-maps]
  (let [revere-mapped-elemn
        (or
         (-> target-categorical-maps column :lookup-table (get element--str))
         (-> target-categorical-maps column :lookup-table (get (keyword element--str)))
         (parse-long element--str)
         (parse-double element--str)
         element--str)]
    (dt/cast revere-mapped-elemn target-datatype)))

(defn- cast-target [prediction-ds column target-datatype target-categorical-maps]
  ;; (println :prediction--meta
  ;;          (-> prediction-ds (tc/head) (get column) meta)
  ;;          :data
  ;;          (-> prediction-ds (tc/head) (get column) seq))
  (->
   (ds/update-column
    prediction-ds
    column
    (fn [col]
      (map #(string->numeric % column target-datatype target-categorical-maps) col)))
   (ds/assoc-metadata [column] :column-type :prediction)))


(defn- post-process-prediction-classification [prediction-ds target-column-name target-datatypes target-categorical-maps]
  (let [renamed-ds (ds/rename-columns prediction-ds {:prediction target-column-name})
        casted-ds (reduce-kv
                   (fn [m k v]
                     (cast-target m k v target-categorical-maps))
                   renamed-ds target-datatypes)
        prob-column-names
        (-> casted-ds
            (ds/drop-columns [target-column-name])
            (ds/column-names))]

    (ds/assoc-metadata casted-ds prob-column-names :column-type :probability-distribution)))

(defn- post-process-prediction-regression [ds target-column-name]
  (-> ds
      (ds/assoc-metadata [:prediction] :column-type :prediction)
      (ds/rename-columns {:prediction target-column-name})))

(defn model->byte-array [model-instance]
  (let [out (java.io.ByteArrayOutputStream.)]
    (.serializeToStream model-instance out)
    (.toByteArray out)))


(defn- train-classification [feature-ds target-ds options]
    ;; (println :target-ts--meta
    ;;          (map meta (-> target-ds vals))
    ;;          :data
    ;;          (map #(take 5 %) (vals target-ds)))
  (let [target-data-type (-> target-ds ds/columns first meta :datatype)

        _ (when (= :object target-data-type)
            (errors/throwf ":object target column not supported"))

        model-instance
        (tribuo/train-classification (make-trainer options)
                                     (ds/append-columns feature-ds (ds/columns target-ds)))]

    {:model-as-bytes
     (model->byte-array model-instance)}))

(defn predict-classification [feature-ds thawed-model {:keys [model-data
                                                              target-columns
                                                              target-categorical-maps
                                                              target-datatypes] :as model}]

  (let [model-instance thawed-model
        target-column-name (first target-columns)
        prediction
        (->
         (tribuo/predict-classification model-instance
                                        feature-ds)
         (post-process-prediction-classification target-column-name target-datatypes target-categorical-maps))]
    (ds/assoc-metadata prediction [target-column-name] :categorical-map (get target-categorical-maps target-column-name))))


(defn- thaw
  [model-data]
  (org.tribuo.Model/deserializeFromStream (java.io.ByteArrayInputStream. (:model-as-bytes model-data))))



(defn evaluate [model]
  (let [feature-ds (-> model :model-data :feature-ds)
        target-ds (-> model :model-data :target-ds)

        model-object (-> model :model-data thaw)
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

  (let [target-ds (-> model :model-data :target-ds)
        target-column-names (ds-mod/inference-target-column-names target-ds)
        _ (assert (< (count  target-column-names) 2) "Can only handle single inference target")

        evaluation (evaluate model)
        predictions (->predictions evaluation)


        regressor-values (-> target-ds (get (first target-column-names)))
        prediction-values (map :prediction predictions)

        target-column-names (ds-mod/inference-target-column-names target-ds)
        regressor-name (name (first target-column-names))
        regressor (Regressor. regressor-name, Double/NaN)]



    (ds/->dataset
     {;; :regressor [ regressor-name]
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


(defn- train-regression [feature-ds target-ds options]
  (let [model-instance (tribuo/train-regression (make-trainer options)
                                                (ds/append-columns feature-ds (ds/columns target-ds)))]

    {:target-ds target-ds
     :feature-ds feature-ds
     :model-as-bytes
     (model->byte-array model-instance)}))

(defn- predict-regression [feature-ds thawed-model {:keys [model-data target-columns]}]
  (let [model thawed-model]
    (->
     (tribuo/predict-regression model feature-ds)
     (post-process-prediction-regression (first target-columns)))))

(defn- safety-first! [values-1 values-2]
  (let [different-types
        (into #{}
              (concat (map type values-1)
                      (map type values-2)))]
    (assert (= 1 (count different-types))
            (format "All values need to be of same type, but found: %s\nvalues-1: %s\nvalues-2: %s"
                    different-types (frequencies values-1) (frequencies values-2)))))



(defn- do-harmonize [ds variable-type filter-fn]
  (let [column (-> ds
                   filter-fn
                   (dscat/reverse-map-categorical-xforms)
                   (ds/columns)
                   first)]
    (assert (some? column)
            (format "No column found matching filter: %s\nmeta of ds columns: %s" 
                    filter-fn 
                    (mapv meta (ds/columns ds)))
            )
    (case variable-type
      :discrete
      (case (-> column meta :datatype)
        :keyword (vec column)
        :int64 (vec column)
        :string (vec column)
        :boolean (vec column)
        :float64 (int-array column))

      :continous (ds-col/to-double-array column)))
  )



(defn- do-harmonize-trueth [ds discrete-or-continous]
  (do-harmonize ds discrete-or-continous cf/target))

(defn- do-harmonize-prediction [ds discrete-or-continous]
  (do-harmonize ds discrete-or-continous cf/prediction))


(defn pre-metric-standardise 
  "converts prediction result and the trueth into either
   seq of 
   :discrete     keyword,string,intXX,...
   :continous   double, float
   or fails.

   `prediction-ds` and `thrueth-ds` are tabular data, 
   usualy of type tech.v3.dataset

   returns map of 
   :prediction (seq)
   :trueth     (seq)

   I case of :discrete the discrete values in :predicion and :trueth
   should have semantically identical meaning, as they might get
   compared via '=' later
   "
  [prediction-ds trueth-ds discrete-or-continous]
  
  {:prediction (do-harmonize-prediction prediction-ds discrete-or-continous)
   :trueth (do-harmonize-trueth trueth-ds discrete-or-continous)}
  )


(defn- score 
  ([model scoring-ds options]
   ;; classificatioon only
   (let [prediction (ml/predict (cf/feature scoring-ds) model)
         trueth (cf/target scoring-ds)
         standardised (pre-metric-standardise prediction trueth :discrete)
         prediction-values (:prediction standardised)
         trueth-values (:trueth standardised)
         _ (safety-first! prediction-values trueth-values)]
     (loss/classification-accuracy prediction-values trueth-values)))
  ([model scoring-ds](score model scoring-ds nil))
  )


(ml/define-model! :scicloj.ml.tribuo/regression
  train-regression
  predict-regression
  {:glance-fn glance-fn-regression
   :augment-fn augment-fn-regression
   :thaw-fn thaw
   :pre-metric-standarisation-fn pre-metric-standardise
   })



(ml/define-model! :scicloj.ml.tribuo/classification
  train-classification
  predict-classification
  {:thaw-fn thaw
   :score-fn score
   :pre-metric-standarisation-fn pre-metric-standardise
   })



;(model-info/register-models train-classification predict-classification train-regression predict-regression)



