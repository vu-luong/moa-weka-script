import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;
import moa.classifiers.meta.AccuracyWeightedEnsemble;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.Example;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.evaluation.LearningCurve;
import moa.streams.ArffFileStream;
import moa.tasks.EvaluatePrequential;

public class TestMoa {
    public static void main(String[] args) {

        WekaToSamoaInstanceConverter wtsic = new WekaToSamoaInstanceConverter();

//        HoeffdingTree ht = new HoeffdingTree();
        AccuracyWeightedEnsemble ht = new AccuracyWeightedEnsemble();
        BasicClassificationPerformanceEvaluator bcpe = new BasicClassificationPerformanceEvaluator();

        ArffFileStream afs = new ArffFileStream();
        afs.arffFileOption.setValue("/Users/AnhVu/Desktop/temp_data/electricity-normalized.arff");
        afs.prepareForUse();


        EvaluatePrequential ep = new EvaluatePrequential();
//        ep.learnerOption.setCurrentObject(ht);
//        ep.streamOption.setCurrentObject(afs);
//        ep.evaluatorOption.setCurrentObject(bcpe);
//        ep.sampleFrequencyOption.setValue(1000);
//        ep.memCheckFrequencyOption.setValue(1000);

//        ep.prepareForUse();
//        System.out.println(ep.doTask());
//        LearningCurve lc = (LearningCurve) ep.doTask();
//        System.out.println(lc);
//        double res = lc.getMeasurement(lc.numEntries() - 1,4);
//        System.out.println(res);

        ht.setModelContext(afs.getHeader());
        System.out.println(afs.getHeader());
        ht.prepareForUse();


        while (afs.hasMoreInstances()) {
            Example trainInst = afs.nextInstance();
            double[] prediction = ht.getVotesForInstance(trainInst);
            System.out.println(trainInst);
            ht.trainOnInstance(trainInst);
        }

    }
}
