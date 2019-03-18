import weka.classifiers.lazy.KStar;
import weka.classifiers.lazy.LWL;
import weka.classifiers.trees.HoeffdingTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;

public class TestWeka {


    public static void main(String[] args) throws Exception {

        // Load input file
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File("/Users/AnhVu/Desktop/temp_data/electricity-normalized.arff"));
        Instances structure = loader.getStructure();
        structure.setClassIndex(structure.numAttributes() - 1);

        // Run the classifier
//        HoeffdingTree model = new HoeffdingTree();
        LWL model = new LWL();
        model.buildClassifier(structure);
        Instance current;
        int cnt = 0;
        while ((current = loader.getNextInstance(structure)) != null) {
            cnt++;
            if (cnt % 100 == 0) {
                System.out.println("Instance: " + cnt);
            }
            if (cnt > 1) {
                double[] prob_array = model.distributionForInstance(current);
            }
            model.updateClassifier(current);
        }


    }

}
