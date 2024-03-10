package org.example;


import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;

/**
 * @author ${USER}
 **/
public class Main {
    public static void main(String[] args) {
//        Model.getModel();
//        Model.train();
//        Model.evaluate();
//        Model.saveModel();

        try{
            MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File("src/main/resources/model.zip"));
            String [] labels = {"Iris setosa", "Iris versicolor", "Iris virginica"};

            INDArray inputs = Nd4j.create(new double [][]{
                    {6.9,3.1,5.1,2.3},
                    {5.1, 3.5, 1.4, 0.2}
            });
            INDArray outputs = model.output(inputs);

            int [] classes = outputs.argMax(1).toIntVector();
            System.out.println(outputs);
            for(int i = 0; i < classes.length; i ++)
                System.out.println("Class: " + labels[classes[i]]);

        }catch(Exception e){
            e.printStackTrace();
        }

    }
}