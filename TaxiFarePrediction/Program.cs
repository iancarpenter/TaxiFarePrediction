using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

using System;
using System.IO;
using System.Threading.Tasks;

namespace TaxiFarePrediction
{
    class Program
    {
        static readonly string _datapath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testdatapath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelpath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static async Task Main(string[] args)
        {
            PredictionModel<TaxiTrip, TaxiTripFarePredition> model = await Train();

            Evaluate(model);

            TaxiTripFarePredition prediction = model.Predict(TestTrips.Trip1);
            Console.WriteLine("Predicted fare: {0}, actual fare: 29.5", prediction.FareAmount);

        }

        public static async Task<PredictionModel<TaxiTrip, TaxiTripFarePredition>> Train()
        {
            var pipeline = new LearningPipeline
            {
                new TextLoader(_datapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ','),

                // fails with Source column "Label" not found without the double brackets
                new ColumnCopier(("FareAmount", "Label")),

                new CategoricalOneHotVectorizer(
                    "VendorId",
                    "RateCode",
                    "PaymentType"),
                new ColumnConcatenator("Features",
                                       "VendorId",
                                       "RateCode",
                                       "PassengerCount",
                                       "TripDistance",
                                       "PaymentType"),
                new FastTreeRegressor()
            };

            PredictionModel<TaxiTrip, TaxiTripFarePredition> model = pipeline.Train<TaxiTrip, TaxiTripFarePredition>();

            await model.WriteAsync(_modelpath);
            return model;
           
        }

        private static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePredition> model)
        {
            var testData = new TextLoader(_testdatapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ',');

            var evaluator = new RegressionEvaluator();
            RegressionMetrics metrics = evaluator.Evaluate(model, testData);

            // Rms is one of the evaluation metrics of the regression model. The lower it is, the better            
            Console.WriteLine($"Rms = {metrics.Rms}");

            // RSquared is another evaluation metric of the regression mode. RSquared takes values between 0 and 1. The 
            // closer its value is to 1, the better the model is.
            Console.WriteLine($"RSquared = {metrics.RSquared}");
        }
    }
}
