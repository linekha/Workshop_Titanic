from pyspark import SparkConf,SparkContext
from pyspark import SQLContext
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer,OneHotEncoder,VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


conf= SparkConf()
sc=SparkContext(conf=conf)
sqlContext = SQLContext(sc)


titanic = sqlContext.read.csv("/user/line66/Titanic/train.csv",header='true',sep=",",inferSchema='true')
titanic_test = sqlContext.read.csv("/user/line66/Titanic/test.csv",header='true',sep=",",inferSchema='true')

titanic.printSchema()
titanic_test.printSchema()

titanic = titanic.fillna("Q",subset='Embarked')
titanic_test = titanic_test.fillna("Q",subset="Embarked")

titanic.first()
titanic_test.first()

total = titanic.count()
print('Titanic Total : {0}'.format(total))

total_test = titanic_test.count()
print('Titanic_test Total:{0}'.format(total_test))

titanic.select(F.col('Embarked')).distinct().count()
titanic_test.select(F.col('Pclass')).distinct().count()



for col in titanic.columns :
  titanic_null = titanic.where(F.col(col).isNull()).count()
  titanic_distinct = titanic.groupby(F.col(col)).agg(F.countDistinct(F.col('Parch'))).show()
  print('Titanic null total : {0}'.format(titanic_null))


for col in titanic_test.columns:
  titanic_test_null = titanic_test.where(F.col(col).isNull()).count()
  titanic_test_distinct = titanic_test.groupby(F.col(col)).agg(F.countDistinct(F.col('Parch'))).show()
  print('Titanic null total : {0}'.format(titanic_test_null))


titanic_grouped = titanic.groupBy(['Pclass','Parch']).agg(F.count('Survived').alias('count_survived'))
titanic_test_grouped = titanic_test.groupBy(['Sex']).agg(sum('Fare').alias('sum_fare'))


def create_dummies_withColumn(df_init,liste):
  df=df_init
  for elmt in liste:
    categories = df_init.select(elmt).distinct().rdd.flatMap(lambda x: x).collect()
    for cat in categories:
      df = df.withColumn(cat,F.when(F.col(elmt) == cat, 1).otherwise(0))
  return df


# Cleanup

def cleanup(df, col_type):
    for col in df.columns:
        tit_total = df.select(col).count()
        tit_isnull = df.select([F.count(
            (F.when((F.col("{0}".format(col))).isNull(), "{0}".format(col))).alias("{0}".format(col)))]).collect()

        int_cols = [item[0] for item in df.dtypes if item[1].startswith('{0}'.format(col_type))]
    for int_col in int_cols:
        tit_mean = df.select(F.mean("{0}".format(int_col))).collect()[0]
        tit_std = df.select(F.stddev("{0}".format(int_col))).collect()[0]
        df = df.withColumn(("{0}".format(int_col)),
                           F.when(F.col("{0}".format(int_col)).isNull(), tit_mean[0] - tit_std[0]).otherwise(
                               F.col("{0}".format(int_col))))
        return df


titanic_dummies = create_dummies_withColumn(titanic,['Embarked','Sex'])
titanic_test_dummies = create_dummies_withColumn(titanic_test,['Embarked','Sex'])

titanic_cleanup = cleanup(titanic,('double','integer'))
titanic_test_cleanup = cleanup(titanic_test,('double','integer'))


(traindf,testdf) = titanic_cleanup.randomSplit([0.7,0.3])

genderIndexer = StringIndexer(inputCol="Sex", outputCol="indexedSex")
embarkIndexer = StringIndexer(inputCol="Embarked", outputCol="indexedEmbarked")

surviveIndexer = StringIndexer(inputCol="Survived", outputCol="indexedSurvived")

# One Hot Encoder on indexed features
genderEncoder = OneHotEncoder(inputCol="indexedSex", outputCol="sexVec")
embarkEncoder = OneHotEncoder(inputCol="indexedEmbarked", outputCol="embarkedVec")

# Create the vector structured data (label,features(vector))
assembler = VectorAssembler(inputCols=["Pclass", "sexVec", "Age", "SibSp", "Fare", "embarkedVec"], outputCol="features")

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedSurvived", featuresCol="features")

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[surviveIndexer, genderIndexer, embarkIndexer, genderEncoder, embarkEncoder, assembler,rf])  # genderIndexer,embarkIndexer,genderEncoder,embarkEncoder,

# Train model.  This also runs the indexers.
model = pipeline.fit(traindf)

# Predictions
predictions = model.transform(testdf)

# Select example rows to display.
predictions.columns

# Select example rows to display.
predictions.select("prediction", "Survived", "features").show(5)

# Select (prediction, true label) and compute test error
predictions = predictions.select(col("Survived").cast("Float"), col("prediction"))
evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))
rfModel = model.stages[6]
print(rfModel)

evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g" % accuracy)

evaluatorf1 = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="f1")
f1 = evaluatorf1.evaluate(predictions)
print("f1 = %g" % f1)

evaluatorwp = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction",metricName="weightedPrecision")
wp = evaluatorwp.evaluate(predictions)
print("weightedPrecision = %g" % wp)

evaluatorwr = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="weightedRecall")
wr = evaluatorwr.evaluate(predictions)
print("weightedRecall = %g" % wr)

