(ns scicloj.ml.linear-regression-test
  (:require
   [clojure.test :as t :refer [is]]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.toydata :as toydata]
   [scicloj.ml.tribuo]
   [tech.v3.dataset :as ds]))

(def diabetes

  (toydata/diabetes-ds))




(t/deftest tidy
  (let [tribuo-linear-sdg
        (ml/train
         diabetes
         {:model-type :scicloj.ml.tribuo/regression
          :tribuo-components [{:name "squared"
                               :type "org.tribuo.regression.sgd.objectives.SquaredLoss"}
                              {:name "trainer"
                               :type "org.tribuo.regression.sgd.linear.LinearSGDTrainer"
                               :properties  {:epochs "100"
                                             :minibatchSize "1"
                                             :objective "squared"}}]
          :tribuo-trainer-name "trainer"})]

  
    (is (= [{:disease-progression 163.65426518599335} {:disease-progression 113.19672792128675} {:disease-progression 156.74746391022254} {:disease-progression 151.2497638618952} {:disease-progression 139.5246990528341}]
           (ds/rows
            (ml/predict (ds/head diabetes) tribuo-linear-sdg))))
    
    
    (t/is (=
           (ds/rows
            (ml/glance tribuo-linear-sdg))
           [{:r.squared 0.2970733043591943 :mae 54.83060340549487 :rmse 64.56217465447965 :rss 1842377.2830830663}]))


    (t/is (=
           [{:age 0.0380759064334241 :sex 0.0506801187398187 :bmi 0.0616962065186885 :bp 0.0218723549949558 :s1 -0.0442234984244464 :s2 -0.0348207628376986 :s3 -0.0434008456520269 :s4 -0.00259226199818282 :s5 0.0199084208763183 :s6 -0.0176461251598052 :disease-progression 151 :.fitted 163.65426518599335 :.resid -12.654265185993353}
            {:age -0.00188201652779104 :sex -0.044641636506989 :bmi -0.0514740612388061 :bp -0.0263278347173518 :s1 -0.00844872411121698 :s2 -0.019163339748222 :s3 0.0744115640787594 :s4 -0.0394933828740919 :s5 -0.0683297436244215 :s6 -0.09220404962683 :disease-progression 75 :.fitted 113.19672792128675 :.resid -38.19672792128675}
            {:age 0.0852989062966783 :sex 0.0506801187398187 :bmi 0.0444512133365941 :bp -0.00567061055493425 :s1 -0.0455994512826475 :s2 -0.0341944659141195 :s3 -0.0323559322397657 :s4 -0.00259226199818282 :s5 0.00286377051894013 :s6 -0.0259303389894746 :disease-progression 141 :.fitted 156.74746391022254 :.resid -15.747463910222535}
            {:age -0.0890629393522603 :sex -0.044641636506989 :bmi -0.0115950145052127 :bp -0.0366564467985606 :s1 0.0121905687618 :s2 0.0249905933641021 :s3 -0.0360375700438527 :s4 0.0343088588777263 :s5 0.0226920225667445 :s6 -0.0093619113301358 :disease-progression 206 :.fitted 151.2497638618952 :.resid 54.75023613810481}
            {:age 0.00538306037424807 :sex -0.044641636506989 :bmi -0.0363846922044735 :bp 0.0218723549949558 :s1 0.00393485161259318 :s2 0.0155961395104161 :s3 0.0081420836051921 :s4 -0.00259226199818282 :s5 -0.0319914449413559 :s6 -0.0466408735636482 :disease-progression 135 :.fitted 139.5246990528341 :.resid -4.524699052834109}
            {:age -0.0926954778032799 :sex -0.044641636506989 :bmi -0.0406959404999971 :bp -0.0194420933298793 :s1 -0.0689906498720667 :s2 -0.0792878444118122 :s3 0.0412768238419757 :s4 -0.076394503750001 :s5 -0.0411803851880079 :s6 -0.0963461565416647 :disease-progression 97 :.fitted 111.3494709834278 :.resid -14.349470983427807}
            {:age -0.0454724779400257 :sex 0.0506801187398187 :bmi -0.0471628129432825 :bp -0.015999222636143 :s1 -0.040095639849843 :s2 -0.0248000120604336 :s3 7.78807997017968E-4 :s4 -0.0394933828740919 :s5 -0.0629129499162512 :s6 -0.0383566597339788 :disease-progression 138 :.fitted 123.99033136526819 :.resid 14.00966863473181}
            {:age 0.063503675590561 :sex 0.0506801187398187 :bmi -0.00189470584028465 :bp 0.0666296740135272 :s1 0.0906198816792644 :s2 0.108914381123697 :s3 0.0228686348215404 :s4 0.0177033544835672 :s5 -0.0358167281015492 :s6 0.00306440941436832 :disease-progression 63 :.fitted 159.22203748830177 :.resid -96.22203748830177}
            {:age 0.0417084448844436 :sex 0.0506801187398187 :bmi 0.0616962065186885 :bp -0.0400993174922969 :s1 -0.0139525355440215 :s2 0.00620168565673016 :s3 -0.0286742944356786 :s4 -0.00259226199818282 :s5 -0.0149564750249113 :s6 0.0113486232440377 :disease-progression 110 :.fitted 156.06779795908577 :.resid -46.067797959085766}
            {:age -0.0709002470971626 :sex -0.044641636506989 :bmi 0.0390621529671896 :bp -0.0332135761048244 :s1 -0.0125765826858204 :s2 -0.034507614375909 :s3 -0.0249926566315915 :s4 -0.00259226199818282 :s5 0.0677363261102861 :s6 -0.0135040182449705 :disease-progression 310 :.fitted 157.27316323150677 :.resid 152.72683676849323}]
           (->
            tribuo-linear-sdg
            (ml/augment diabetes)
            (ds/head 10)
            (ds/rows))))))


(def model-specs

  [{:model-type :scicloj.ml.tribuo/regression
    :tribuo-components [{:name "squaredloss"
                         :type "org.tribuo.regression.sgd.objectives.SquaredLoss"}
                        {:name "trainer"
                         :type "org.tribuo.regression.sgd.linear.LinearSGDTrainer"
                         :properties  {:epochs "100"
                                       :minibatchSize "1"
                                       :objective "squaredloss"}}]
    :tribuo-trainer-name "trainer"}


   {:model-type :scicloj.ml.tribuo/regression
    :tribuo-components [{:name "huber"
                         :type "org.tribuo.regression.sgd.objectives.Huber"
                         :properties {:cost "10"}}
                        {:name "adagrad"
                         :type "org.tribuo.math.optimisers.AdaGrad"
                         :properties {:initialLearningRate "1.0"
                                      :epsilon "1e-6"
                                      :initialValue "0.0"}}
                        {:name "trainer"
                         :type "org.tribuo.regression.sgd.linear.LinearSGDTrainer"
                         :properties  {:epochs "50"
                                       :minibatchSize "1"
                                       :optimiser "adagrad"
                                       :objective "huber"}}]
    :tribuo-trainer-name "trainer"}])

(t/deftest test-two-specs
  (ml/train diabetes (first model-specs))
  (ml/train diabetes (second model-specs))
  )

