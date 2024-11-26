(ns scicloj.ml.tribuo.model-info
  (:require
   [clojure.java.classpath]
   [clojure.reflect]
   [clojure.string :as str]
   [scicloj.metamorph.ml :as ml])
  (:import
   [com.oracle.labs.mlrg.olcut.config DescribeConfigurable])
  )

(defn all-configurables[interface]

  (->> (clojure.java.classpath/classpath-jarfiles)
       (filter (fn [^java.util.jar.JarFile jf]
                 (re-matches #".*tribuo.*" (.getName jf))))
       (mapcat clojure.java.classpath/filenames-in-jar)
       (map (fn [class-filename]
              (try (some-> class-filename
                           (str/replace #"/" ".")
                           (str/replace #"\.class$" "")
                           (Class/forName))
                   (catch Exception _ nil))))
       (filter (fn [cls]
                 (->> cls
                      supers
                      (some #(= % interface 
                                ;org.tribuo.Trainer
                                ;com.oracle.labs.mlrg.olcut.config.Configurable
                                
                                )))))))


(defn- configurable->docu [class]
  (->>
   (DescribeConfigurable/generateFieldInfo class)
   vals
   (map (fn [field-info]
          {:name  (.name field-info)
           :description (.description field-info)
           :type (.getGenericType (.field field-info))
           :default (.defaultVal field-info)}))))


(defn- safe-configurable->docu [class]
  {:class class
   :options
   (try
     (configurable->docu class)
     (catch Exception _ nil))})

(defn- trainer-infos []
  (->> (all-configurables org.tribuo.Trainer)
       (map safe-configurable->docu)
       (remove #(empty? (:options %)))))


(defn- class->tribuo-url [class]
  (if (nil? class)
    ""
    (str "https://tribuo.org/learn/4.3/javadoc/"
         (str/replace (.getName class)
                      "." "/")
         ".html")))

(defn- train-wrapper [trainer-class train-fn]
  (fn [feature-ds target-ds options]
    (train-fn feature-ds target-ds
              (assoc options :trainer-class-name trainer-class))))

(defn register-models [train-classification predict-classification train-regression predict-regression ]
  (run!
   (fn [trainer-info]
     (let [opts {:options (:options trainer-info)
                 :documentation {:javadoc (class->tribuo-url (:class trainer-info))}}
           fq-class-name (.getName (:class trainer-info))

           name-pieces (str/split fq-class-name #"\.")
           type (nth name-pieces 2)
           trainer-name   (last name-pieces) ;(-> name-pieces last (str/replace "Trainer" "") csk/->kebab-case)
           
           [train-fn predict-fn]
           (cond
             (str/starts-with? fq-class-name "org.tribuo.classification")
             [train-classification predict-classification]
             (str/starts-with? fq-class-name "org.tribuo.regression")
             [train-regression predict-regression]
             :else nil)]
       (when (some? train-fn)
         (ml/define-model! (keyword "scicloj.ml.tribuo" (format "%s.%s" type trainer-name))
           (train-wrapper fq-class-name train-fn)
           predict-fn
           opts))))
   (trainer-infos)))
