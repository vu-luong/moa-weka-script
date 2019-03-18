import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVSaver;

import java.io.*;
import java.nio.file.Paths;
import java.util.UUID;

public class ConvertToCSV {

    public static void main(String[] args) throws IOException {
        BufferedReader reader =
                new BufferedReader(new FileReader("/Users/AnhVu/Desktop/temp_data/electricity-normalized.arff"));
        ArffLoader.ArffReader arff = new ArffLoader.ArffReader(reader);
        Instances data = arff.getData();
        data.setClassIndex(data.numAttributes() - 1);

        BufferedWriter writer = new BufferedWriter(new FileWriter("/Users/AnhVu/Desktop/temp_data/converted/res.csv"));

        for (int i = 0; i < data.numInstances(); i++) {
            for (int j = 0; j < data.numAttributes() - 1; j++) {
                writer.write(data.get(i).value(j) + ",");
            }
            writer.write((int) data.get(i).classValue() + "\n");
        }

        writer.close();

    }

}
