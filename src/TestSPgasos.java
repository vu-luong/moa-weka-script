import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;
import moa.classifiers.functions.SPegasos;
import moa.classifiers.meta.AccuracyWeightedEnsemble;
import moa.classifiers.meta.HeterogeneousEnsembleBlastFadingFactors;
import moa.classifiers.meta.WEKAClassifier;
import moa.classifiers.trees.HoeffdingAdaptiveTree;
import moa.core.DoubleVector;
import moa.core.Utils;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class TestSPgasos {

    public static void main(String[] args) throws IOException {
        BufferedReader reader =
                new BufferedReader(new FileReader("/Users/AnhVu/Desktop/temp_data/electricity-normalized.arff"));
        ArffLoader.ArffReader arff = new ArffLoader.ArffReader(reader);
        Instances data = arff.getData();
        data.setClassIndex(data.numAttributes() - 1);

        WekaToSamoaInstanceConverter wtsic = new WekaToSamoaInstanceConverter();

        HoeffdingAdaptiveTree ht = new HoeffdingAdaptiveTree();

//        HeterogeneousEnsembleBlastFadingFactors ht = new HeterogeneousEnsembleBlastFadingFactors();

//        SPegasos ht = new SPegasos();
//        ht.lossFunctionOption.setChosenLabel("LOGLOSS");
//        ht.setLossFunction(0);

//        AccuracyWeightedEnsemble ht = new AccuracyWeightedEnsemble();
//        ht.learnerOption.setValueViaCLIString("meta.WEKAClassifier -l (weka.classifiers.functions.SMO -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\")");
//        ht.chunkSizeOption.setValue(1000);

        ht.prepareForUse();
        ht.setModelContext(new InstancesHeader(wtsic.samoaInstances(data)));


//        System.out.println(ht.learnerOption.getValueAsCLIString());

        double error = 0;
        for (int i = 0; i < data.numInstances(); i++) {
            if (i % 1000 == 0) {
                System.out.println("instance: " + i);
            }

            Instance currentInstance = wtsic.samoaInstance(data.get(i));
            double[] prediction = ht.getVotesForInstance(currentInstance);
            double[] randomPrediction = {0.5, 0.5};
            DoubleVector v = new DoubleVector(prediction);
            v.addValues(new double[currentInstance.numClasses()]); //Add zeros
            if (v.sumOfValues() > 0.0) {
                v.normalize();
            } else {
                v = new DoubleVector(randomPrediction);
            }
//            System.out.println(v);

            int yPred = Utils.maxIndex(prediction);

            int trueClass = (int) currentInstance.classValue();
            if (yPred != trueClass) {
                error += 1;
            }
            ht.trainOnInstance(currentInstance);
        }
        System.out.println(error + "");
        System.out.println(data.numInstances() + "");

        error = error / data.numInstances();
        System.out.println("Error = " + error);

    }
}
