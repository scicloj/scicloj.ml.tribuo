[![CI](https://github.com/scicloj/scicloj.ml.tribuo/actions/workflows/clojure.yml/badge.svg)](https://github.com/scicloj/scicloj.ml.tribuo/actions/workflows/clojure.yml)

# scicloj/scicloj.ml.tribuo

Integration of [tribuo](https://tribuo.org) ML library into the scicloj.ml / metamorph framework.

## Usage

All models of tribuo are supported out of the box, via a configuration map. See tribuo website for details.


``` clojure
(require '[scicloj.ml.tribuo]
          '[scicloj.metamorph.ml :as ml]
          '[tech.v3.dataset.modelling :as dsmod]
          '[scicloj.metamorph.ml.toydata :as data]
          )
(def split (dsmod/train-test-split iris ))
(def model (ml/train (:train-ds split)
                        {:model-type :scicloj.ml.tribuo/classification
                         :tribuo-components [{:name "trainer"
                                              :type "org.tribuo.classification.dtree.CARTClassificationTrainer"}]
                         :tribuo-trainer-name "trainer"}))
(def prediction (ml/predict (:test-ds split) model))
```

the same as above , using metamorh pipeline which can encapsulate the state

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
(->> (mm/fit-pipe (:train-ds split) cart-pipeline)
     (mm/transform-pipe (:test-ds split) cart-pipeline)
     :metamorph/data)

```

## build and deploy

Run the project's tests (they'll fail until you edit them):

    $ clojure -T:build test

Run the project's CI pipeline and build a JAR (this will fail until you edit the tests to pass):

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
