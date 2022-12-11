
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;



public class predictionModel {

	/* This function will predict the evaluation and performance of data set using giving input*/
    public void predictionDataSetPerformance(Dataset<Row> test_data_set){
        RegressionEvaluator EvaluatorModel = new RegressionEvaluator().setLabelCol("quality").setPredictionCol("prediction").setMetricName("mae");
        
       // Get the Result of the mean 
        double meanDataSet = EvaluatorModel.evaluate(test_data_set);
        System.out.println("Mean Data Set: " +meanDataSet);
    }
    
	
	public static void main(String[] args) {
		
		
		Logger.getLogger("org").setLevel(Level.ERROR);
		if (args.length < 2) {
			System.out.println("test file and model path are missing.");
			System.exit(1);
		}

	
		String test_file = args[0];
		String saved_model = args[1];

		// Establishing New Spark Session
		SparkSession newSparkSession = new SparkSession.Builder().appName("Wine Quality Prediction Model").getOrCreate();
		

		ApplicationPredictionModel applicationpredictionmodeltrainer = new ApplicationPredictionModel();
		PredictionModel modelPrediction = new PredictionModel();
		
		
		PipelineModel pipelineModel = pipelineModel.load(saved_model);
		
	
		Dataset<Row> dataSetTest = applicationpredictionmodeltrainer.fetchAndConvertDataSet(newSparkSession, test_file);
		

		Dataset<Row> applicationpredictedDataSet = pipelineModel.transform(dataSetTest);
		
		
		applicationpredictedDataSet.show();
		
		/* Evaluation and performance of data set */
		modelPrediction.predictionEvaluationDataSetPerformance(applicationpredictedDataSet);		
	}
}

