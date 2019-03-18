import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;
import moa.classifiers.functions.MajorityClass;
import moa.classifiers.functions.NoChange;
import moa.classifiers.functions.SGD;
import moa.classifiers.functions.SPegasos;
import moa.classifiers.lazy.kNN;
import moa.classifiers.lazy.kNNwithPAW;
import moa.classifiers.meta.AccuracyWeightedEnsemble;
import moa.classifiers.meta.HeterogeneousEnsembleBlastFadingFactors;
import moa.classifiers.trees.ASHoeffdingTree;
import moa.classifiers.trees.DecisionStump;
import moa.classifiers.trees.HoeffdingAdaptiveTree;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.DoubleVector;
import moa.core.Utils;
import weka.classifiers.lazy.KStar;
import weka.classifiers.lazy.LWL;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class FromWeka {

    public static void main(String[] args) throws IOException {
        BufferedReader reader =
                new BufferedReader(new FileReader("/Users/AnhVu/Desktop/temp_data/electricity-normalized.arff"));
        ArffReader arff = new ArffReader(reader);
//        System.out.println(arff.getStructure());
        Instances data = arff.getData();
//        System.out.println(data.stringFreeStructure());
        data.setClassIndex(data.numAttributes() - 1);

        WekaToSamoaInstanceConverter wtsic = new WekaToSamoaInstanceConverter();

//        HoeffdingTree ht = new HoeffdingTree();
//        SGD ht = new SGD();
//        HoeffdingOptionTree hot = new HoeffdingOptionTree();
//        RuleClassifier rc = new RuleClassifier();

//        NaiveBayes nb = new NaiveBayes();
//        RandomHoeffdingTree rht = new RandomHoeffdingTree();

//        WEKAClassifier wc = new WEKAClassifier();
//        wc.baseLearnerOption.setValueViaCLIString("weka.classifiers.trees.J48 -C 0.25 -M 2");
//        wc.prepareForUse();
//        kNN ht = new kNN();
//        ht.kOption.setValue(1);

//        ASHoeffdingTree ht = new ASHoeffdingTree();
//        kNN ht = new kNN();
//        ht.kOption.setValue(1);

//        AccuracyWeightedEnsemble ht = new AccuracyWeightedEnsemble();
//        ht.learnerOption.setValueViaCLIString("meta.WEKAClassifier -l weka.classifiers.trees.DecisionStump");
//        ht.chunkSizeOption.setValue(1000);

//        LWL lwl = new LWL();
        HeterogeneousEnsembleBlastFadingFactors ht = new HeterogeneousEnsembleBlastFadingFactors();

        ht.prepareForUse();
        ht.setModelContext(new InstancesHeader(wtsic.samoaInstances(data)));

//        System.out.println(ht.getSubClassifiers().length + "");
//        ht.setModelContext();

        long startTime = System.currentTimeMillis();
        com.yahoo.labs.samoa.instances.Instances vu = wtsic.samoaInstances(data);
        System.out.println(data.get(4));
        System.out.println(vu.get(4));

        if (1 == 1) return;

        double error = 0;
        for (int i = 0; i < data.numInstances(); i++) {
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
            if (i % 1000 == 0) System.out.println(v);
//            System.out.println(prediction[0] + "-");

            int yPred = Utils.maxIndex(prediction);
//            System.out.println((i + 1) + ": " + (yPred + 1) + ", " + prediction[0]);
            int trueClass = (int) currentInstance.classValue();
            if (yPred != trueClass) {
                error += 1;
            }
//            System.out.println("" + currentInstance);
            ht.trainOnInstance(currentInstance);
        }
        System.out.println(error + "");
        System.out.println(data.numInstances() + "");

        error = error / data.numInstances();
        System.out.println("Error = " + error);
        long elapsedTime = System.currentTimeMillis() - startTime;

        System.out.println("Time: " + (elapsedTime / 1000));
//        System.out.println("Accuracy = " + (1 - error));

//        MajorityClass mc = new MajorityClass();
//        NoChange nc = new NoChange();
//        SPegasos sp = new SPegasos();
//        sp.lossFunctionOption.setChosenLabel("HINGE");
//
//        SGD sgd = new SGD();
//        sgd.lossFunctionOption.setChosenLabel("HINGE");
//
//        kNN knn = new kNN();
//        knn.kOption.setValue(1);
//
//        kNNwithPAW kwp = new kNNwithPAW();
//        DecisionStump ds = new DecisionStump();
//
//        ASHoeffdingTree asht = new ASHoeffdingTree();
//        HoeffdingAdaptiveTree hat = new HoeffdingAdaptiveTree();
//
//        AccuracyWeightedEnsemble awe = new AccuracyWeightedEnsemble();
//        awe.learnerOption.setValueViaCLIString("meta.WEKAClassifier -l (weka.classifiers.functions.SMO -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\")");

    }
}
