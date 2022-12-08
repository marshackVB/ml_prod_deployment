from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pyspark.sql.functions as func

spark = SparkSession.builder.getOrCreate()


def compute_passenger_ticket_features(df):
  
            # Extract characters of ticket if they exist
  return (df.withColumn('TicketChars_extract', func.regexp_extract(col('Ticket'), '([A-Za-z]+)', 1))
             .selectExpr("*", "case when length(TicketChars_extract) > 0 then upper(TicketChars_extract) else NULL end as TicketChars")
             .drop("TicketChars_extract")
          
            # Extract the Cabin character
             .withColumn("CabinChar", func.split(col("Cabin"), '')[0])
          
            # Indicate if multiple Cabins are present
             .withColumn("CabinMulti_extract", func.size(func.split(col("Cabin"), ' ')))
             .selectExpr("*", "case when CabinMulti_extract < 0 then '0' else cast(CabinMulti_extract as string) end as CabinMulti")
             .drop("CabinMulti_extract")
          
            # Round the Fare column
             .withColumn("FareRounded", func.round(col("Fare"), 0))
         
             .drop('Ticket', 'Cabin'))
  

def compute_passenger_demographic_features(df):
  
             # Extract prefic from name, such as Mr. Mrs., etc.
  return (df.withColumn('NamePrefix', func.regexp_extract(col('Name'), '([A-Za-z]+)\.', 1))
             # Extract a secondary name in the Name column if one exists
            .withColumn('NameSecondary_extract', func.regexp_extract(col('Name'), '\(([A-Za-z ]+)\)', 1))
             # Create a feature indicating if a secondary name is present in the Name column
            .selectExpr("*", "case when length(NameSecondary_extract) > 0 then NameSecondary_extract else NULL end as NameSecondary")
            .drop('NameSecondary_extract')
            .selectExpr("PassengerId",
                        "Name",
                        "Sex",
                        "case when Age = 'NaN' then NULL else Age end as Age",
                        "SibSp",
                        "NamePrefix",
                        "NameSecondary",
                        "case when NameSecondary is not NULL then '1' else '0' end as NameMultiple"))
  
  
  