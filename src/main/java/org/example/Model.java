package org.example;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

/**
 * @author Eren
 **/
public class Model {
    private static final double learningRate = 0.001;
    private static final int features = 4;
    private static final int outputs = 10;
    private static final int classes = 3;
    private static MultiLayerConfiguration configuration;
    private static MultiLayerNetwork model;
    private static boolean isTrained = false;
    private static void initConfiguration(){
       configuration =  new NeuralNetConfiguration.Builder()
                .updater(new Adam(learningRate))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(features).nOut(outputs).activation(Activation.SIGMOID).build())
                .layer(1, new OutputLayer.Builder().nIn(outputs).nOut(classes)
                        .lossFunction(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();
    }
    private static MultiLayerConfiguration getConfiguration(){
        if(configuration == null)
            initConfiguration();
        return configuration;
    }
    private static void initModel(){
        if(configuration == null) initConfiguration();
        System.out.println("---------------------------INIT MODEL: 0%--------------------------");
        model = new MultiLayerNetwork(getConfiguration());
        model.init();
        System.out.println("---------------------------INIT MODEL: 100%------------------------");
    }
    public static MultiLayerNetwork getModel(){
        if(model == null)
            initModel();
        return model;
    }
    public static void printConfiguration(){
        System.out.println(getConfiguration().toJson());
    }
    public static void train(){
        if(isTrained)
            return;
        System.out.println("----------------------------MODEL'S TRAINING: 0%-------------------");
        try {
            File fileTrain = new ClassPathResource("data/iris-train.csv").getFile();
            RecordReader recordReaderTrain = new CSVRecordReader();
            recordReaderTrain.initialize(new FileSplit(fileTrain));
            DataSetIterator dataSetIteratorTrain =
                    new RecordReaderDataSetIterator(recordReaderTrain, 1, 4, 3);

            /*while(dataSetIteratorTrain.hasNext()){
                DataSet dataSet = dataSetIteratorTrain.next();
                System.out.println(dataSet.getFeatures());
                System.out.println(dataSet.getLabels());
            }*/
            int epochs = 100;

            for (int i = 0; i < epochs; i++)
                model.fit(dataSetIteratorTrain);
        }catch(Exception e){
            e.printStackTrace();
        }
        isTrained = true;
        System.out.println("----------------------------MODEL'S TRAINING: 100%-----------------");
    }

    public static void evaluate(){
        try{
            System.out.println("----------------------------MODEL'S EVALUATION: 0%-----------------");
            File fileTest = new ClassPathResource("data/iris-test.csv").getFile();
            RecordReader recordReaderTest = new CSVRecordReader();
            recordReaderTest.initialize(new FileSplit(fileTest));
            DataSetIterator dataSetIteratorTest =
                    new RecordReaderDataSetIterator(recordReaderTest, 1, 4, 3);
            Evaluation evaluation = new Evaluation();
            while(dataSetIteratorTest.hasNext()){
                DataSet dataSet = dataSetIteratorTest.next();
                INDArray features = dataSet.getFeatures();
                INDArray targetLabels = dataSet.getLabels();
                INDArray predictedLabels = model.output(features);
                evaluation.eval(predictedLabels, targetLabels);
            }
            System.out.println(evaluation.stats());
            System.out.println("----------------------------MODEL'S EVALUATION: 100%---------------");
        }catch(Exception e){
            e.printStackTrace();
        }
    }

    public static void testModel(){

        System.out.println("----------------------------MODEL'S TEST: 0%-----------------------");
        INDArray inputData = Nd4j.create(new double [][]{
                {5.1, 3.5, 1.4, 0.2},
                {5.1, 3.5, 1.4, 0.2},
                {5.1, 3.5, 1.4, 0.2},
                {5.1, 3.5, 1.4, 0.2}
        });
        INDArray output = model.output(inputData);
        System.out.println(output);
        int[] classes = output.argMax(1).toIntVector();
        for(int i = 0; i < classes.length; i ++)
            System.out.println("Class: " + classes[i]);
        System.out.println("----------------------------MODEL'S TEST: 100%---------------------");
    }
    public static void saveModel(){
        try{
            ModelSerializer.writeModel(model, "src/main/resources/model.zip", true);
            System.out.println("----------------------------MODEL HAS BEEN SAVED-------------------");
        }catch(Exception e){
            e.printStackTrace();
        }
    }
}
