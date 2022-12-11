
	import java.io.IOException;
	import java.util.HashMap;
	import org.apache.log4j.Level;
	import org.apache.log4j.Logger;
	import org.apache.spark.ml.feature.VectorAssembler;
	import org.apache.spark.ml.regression.LinearRegression;
	import org.apache.spark.ml.Pipeline;
	import org.apache.spark.ml.PipelineModel;
	import org.apache.spark.ml.PipelineStage;
	import org.apache.spark.sql.SparkSession;
	import org.apache.spark.sql.Dataset;
	import org.apache.spark.sql.Row;

	public class applicationPredictionModel  {
		
	    // This function will fetch the data and convert the given data set
		
	    public Dataset <Row> fetchDataSet(SparkSession sparkSession, String csvFileName){
	    	
	        HashMap<String, String> csvFileList = new HashMap<String,String>();
	        csvFileList.put("header", "true");
	        csvFileList.put("inferSchema", "true");   
	        csvFileList.put("delimiter",";");
	            
	       
	        Dataset<Row> row_data_set = sparkSession.read().options(csvFileList).csv(csvFileName);
	        
	      
	        Dataset<Row> clean_data_set = row_data_set.dropDuplicates();
	        
	        
	        Dataset<Row> final_Data_Set = convertDataSet(clean_data_set);
	        return final_Data_Set;
	    }
	    
	 
	    public Dataset<Row> convertDataSet(Dataset<Row> newDataSet){
	    	
	    
	    	Dataset<Row> featureColumns = newDataSet.select("fixed acidity", "volatile acidity", "citric acid", "residual sugar",
	                "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol");
	        
	    
	    	VectorAssembler vectorAssembler = new VectorAssembler().setInputCols(featureColumns.columns()).setOutputCol("features");
	        
	        return convertDataSet;
	    }
	    
	
	    public static void main(String[] args) throws IOException {
	        
	    
	    	Logger.getLogger("org").setLevel(Level.ERROR);
	    	
	        if (args.length < 3) {
	            System.err.println("Training data set, validation data set, modal path are missing");
	            System.exit(1);
	        }

	     
	        String training_data_set = args[0];
	        String validation_data_set = args[1];
	        String saved_model_data_set = args[2];

	        ApplicationPredictionModel applipred = new ApplicationPredictionModel();
	        PredictionModel predictModel = new PredictionModel();
	        
	        SparkSession newSparkSession = new SparkSession.Builder().appName("Wine Quality Application Prediction model").getOrCreate();
	        
	        
	        Dataset<Row> final_data_set = applipred.fetchAndConvertDataSet(newSparkSession, training_data_set);        
	        
	  
	        LinearRegression linearRegressionModel = new LinearRegression().setMaxIter(50).setRegParam(0).setFeaturesCol("features").setLabelCol("quality");        
	        
	        
	        Pipeline pl = new Pipeline().setStages(new PipelineStage[]{linearRegressionModel});
	        PipelineModel pipeline_model = pipeline.fit(final_data_set);
	        
	        // fetching and converting the data set and removing the duplication in row */
	        Dataset<Row> validation_data = applipred.fetchAndConvertDataSet(newSparkSession, validation_data_set);
	        
	        //  Pipeline will read the machine learning model from the given input */
	        Dataset<Row> predicted_data = pipeline_model.transform(validation_data);
	        
	       
	        predicted_data.show();
	       
	      
	        predictModel.predictionEvaluationDataSetPerformance(predicted_data);
	        pipeline_model.write().overwrite().save(saved_model_data_set);
	    }
	}

}
