(ns scicloj.ml.tribuo
  (:require
   [scicloj.metamorph.ml :as ml]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.datatype :as dt]
   [tech.v3.datatype.functional :as fun]
   [fastmath.stats :as stats]
   [tech.v3.datatype.errors :as errors]
   [tech.v3.libs.tribuo :as tribuo]
   [tablecloth.api :as tc])
  (:import [org.tribuo.regression.evaluation RegressionEvaluator]
          [org.tribuo.regression Regressor]))

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
  (println :prediction--meta
           (-> prediction-ds (tc/head) (get column) meta)
           :data
           (-> prediction-ds (tc/head) (get column) seq))
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
                   renamed-ds target-datatypes)]
    casted-ds))

(defn- post-process-prediction-regression [ds target-column-name]
  (-> ds
      (ds/assoc-metadata [:prediction] :column-type :prediction)
      (ds/rename-columns {:prediction target-column-name})))




(ml/define-model! :scicloj.ml.tribuo/classification

  (fn [feature-ds target-ds options]

    ;; (println :target-ts--meta
    ;;          (map meta (-> target-ds vals))
    ;;          :data
    ;;          (map #(take 5 %) (vals target-ds)))



    (let [target-data-type (-> target-ds ds/columns first meta :datatype)]

      (when (= :object target-data-type)
        (errors/throwf ":object target column not supported"))

      {:model-instance
       (tribuo/train-classification (make-trainer options)
                                    (ds/append-columns feature-ds (ds/columns target-ds)))}))


  (fn [feature-ds thawed-model {:keys [model-data
                                       target-columns
                                       target-categorical-maps
                                       target-datatypes] :as model}]

    (let [model-instance (:model-instance model-data)
          target-column-name (first target-columns)
          prediction
          (->
           (tribuo/predict-classification model-instance
                                          feature-ds)
           (post-process-prediction-classification target-column-name target-datatypes target-categorical-maps))]
      (ds/assoc-metadata prediction [target-column-name] :categorical-map (get target-categorical-maps target-column-name))))
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
       (post-process-prediction-regression (first target-columns)))))
  {:glance-fn glance-fn-regression
   :augment-fn augment-fn-regression})


(comment

  (import '[com.oracle.labs.mlrg.olcut.config DescribeConfigurable]
          '[com.oracle.labs.mlrg.olcut.config Configurable]
          '[org.tribuo.regression.sgd.linear LinearSGDTrainer])


  (defn configurable->docu [class]
    (->>
     (DescribeConfigurable/generateFieldInfo class)
     vals
     (map (fn [field-info]
            (def field-info field-info)
            (map :name
                 (:members (clojure.reflect/reflect field-info)))
            {:name  (.name field-info)
             :description (.description field-info)
             :type (.getGenericType (.field field-info))
             :default (.defaultVal field-info)}))))

  (def tribuo-trainers
    ["org.tribuo.regression.liblinear.LibLinearRegressionTrainer"
     "org.tribuo.regression.liblinear.LinearRegressionType"
     "org.tribuo.classification.ensemble.AdaBoostTrainer"
     "org.tribuo.classification.dtree.CARTClassificationTrainer"])
  

  (defn safe-class-for-name [s]
    (try
      (Class/forName s)
      (catch Exception e nil)))
  

  (def trainer-classes
    (->>
     (map safe-class-for-name tribuo-trainers)
     (remove nil?)))

  (map
   configurable->docu
   trainer-classes)
  )

