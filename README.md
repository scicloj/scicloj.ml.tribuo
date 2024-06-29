[![Clojars Project](https://img.shields.io/clojars/v/org.scicloj/scicloj.ml.tribuo.svg)](https://clojars.org/org.scicloj/scicloj.ml.tribuo)
[![CI](https://github.com/scicloj/scicloj.ml.tribuo/actions/workflows/clojure.yml/badge.svg)](https://github.com/scicloj/scicloj.ml.tribuo/actions/workflows/clojure.yml)

![ml logo](https://github.com/scicloj/graphic-design/blob/live/icons/scicloj.ml.svg)

# scicloj/scicloj.ml.tribuo

Integration of [tribuo](https://tribuo.org) ML library into the scicloj.ml / metamorph framework.

## Usage

All models of tribuo are supported out of the box, via a configuration map. See tribuo website for details.
Tribuos distributes its models in different jars. When using a certain model, you need to add therefore the correct maven dependency which contains
the model you want to use. This [table](https://github.com/oracle/tribuo/blob/main/docs/PackageOverview.md) contains an overview which model is in which maven artifcat. See the `test` alias [here](https://github.com/scicloj/scicloj.ml.tribuo/blob/2744f3ffff2f90ce7464f8f3850a0714b952fa5d/deps.edn#L19) wich does this setup.



Prepare data:
```clojure
(require '[scicloj.ml.tribuo]
          '[scicloj.metamorph.ml :as ml]
          '[tech.v3.dataset.modelling :as dsmod]
          '[scicloj.metamorph.ml.toydata :as data]
          )

(def iris (data/iris-ds))
(def split (dsmod/train-test-split iris ))

```

train / predict:
``` clojure
(def model (ml/train (:train-ds split)
                        {:model-type :scicloj.ml.tribuo/classification
                         :tribuo-components [{:name "trainer"
                                              :type "org.tribuo.classification.dtree.CARTClassificationTrainer"}]
                         :tribuo-trainer-name "trainer"}))
(def prediction (ml/predict (:test-ds split) model))
```

the same as above , using a metamorh pipeline which can encapsulate the state (= the trained model in this case)

``` clojure
;;  the same using metamorh pipelines

(require '[scicloj.metamorph.core :as mm]')
(def cart-pipeline
  (mm/pipeline
   (ml/model {:model-type :scicloj.ml.tribuo/classification
              :tribuo-components [{:name "trainer"
                                   :type "org.tribuo.classification.dtree.CARTClassificationTrainer"}]
              :tribuo-trainer-name "trainer"})))


;;  no global variable needed, as state is in context
(->> (mm/fit-pipe (:train-ds split) cart-pipeline)           ;train model
     (mm/transform-pipe (:test-ds split) cart-pipeline)      ;make prediction  
     :metamorph/data)                                        ;extract prediction from context

```

## build and deploy

Run the project's tests 

    $ clojure -T:build test

Run the project's CI pipeline and build a JAR: 

    $ clojure -T:build ci

This will produce an updated `pom.xml` file with synchronized dependencies inside the `META-INF`
directory inside `target/classes` and the JAR in `target`. You can update the version (and SCM tag)
information in generated `pom.xml` by updating `build.clj`.

Install it locally (requires the `ci` task be run first):

    $ clojure -T:build install

Deploy it to Clojars -- needs `CLOJARS_USERNAME` and `CLOJARS_PASSWORD` environment
variables (requires the `ci` task be run first):

    $ clojure -T:build deploy

Your library will be deployed to net.clojars.scicloj/scicloj.ml.tribuo on clojars.org by default.

## License

Copyright © 2024 Carsten Behring

Distributed under the Eclipse Public License version 1.0.
