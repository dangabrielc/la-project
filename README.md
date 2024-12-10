Proiect LA - Ciubotaru Dan Gabriel.

Acest proiect a avut ca scop analizarea datelor legate de crimele care au avut loc in America in decursul ultimilor ani.
Pentru acest proiect am folosit urmatoarele librarii:
* pyspark - SparkSession
          - pyspark.sql.functions - col, to_date, year, monh, count si lit.
          - pyspark.ml.features - VectorAssembler si StringIndexer
          - pyspark.ml.regression - RandomForestRegressor
  
* matplotlib - matlabplot.pyplot
* seaborn
* sklearn.preprocessing - LabelEncoder.

In acest proiect am folosit datele pentru a analiza urmatoarele cazuri:
* Distributia crimelor pe an
* Distributia crimelor pe luna
* Cele mai intalnite 10 tipuri de crime
* Cele mai multe crime per locatie in lunile recente
* Predictie a crimelor pe luna pentru urmatorul an
* Matricea de corelatie
