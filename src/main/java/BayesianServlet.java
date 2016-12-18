import weka.classifiers.bayes.net.EditableBayesNet;
import weka.classifiers.bayes.net.MarginCalculator;
import weka.core.Instances;
import weka.core.SerializedObject;
import weka.core.converters.ArffLoader;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * Created by sbreban on 12/18/16.
 */
public class BayesianServlet extends HttpServlet {

  @Override
  protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
    PrintWriter printWriter = resp.getWriter();
    String temperature = req.getParameter("temperature");
    String windy = req.getParameter("windy");
    String outlook = req.getParameter("outlook");
    String snow_cover = req.getParameter("snow_cover");
    String rainfall = req.getParameter("rainfall");

    try {
      String fileName = "/ski.arff";
      System.out.println("\nReading file " + fileName + "...");
      ArffLoader loader = new ArffLoader();
      if (fileName.startsWith("http:") || fileName.startsWith("ftp:"))
        loader.setURL(fileName);
      else {
        String path = getServletContext().getRealPath(fileName);
        loader.setSource(new File(path));
      }
      Instances data = loader.getDataSet();

      System.out.println("\nHeader of dataset:\n");
      System.out.println(new Instances(data, 0));

      EditableBayesNet bayesNet = new EditableBayesNet(data);
      bayesNet.buildClassifier(data);

      MarginCalculator m_marginCalculator = new MarginCalculator();
      m_marginCalculator.calcMargins(bayesNet);
      SerializedObject so = new SerializedObject(m_marginCalculator);
      MarginCalculator m_marginCalculatorWithEvidence = (MarginCalculator) so.getObject();
      for (int iNode = 0; iNode < bayesNet.getNrOfNodes(); iNode++) {
        if (bayesNet.getEvidence(iNode) >= 0) {
          m_marginCalculatorWithEvidence.setEvidence(iNode,
              bayesNet.getEvidence(iNode));
        }
      }
      for (int iNode = 0; iNode < bayesNet.getNrOfNodes(); iNode++) {
        bayesNet.setMargin(iNode,
            m_marginCalculatorWithEvidence.getMargin(iNode));
      }

      for (int iNode = 0; iNode < bayesNet.getNrOfNodes(); iNode++) {
        String[] values = bayesNet.getValues(iNode);
        double[] margins = bayesNet.getMargin(iNode);
        for (int i = 0; i < values.length; i++) {
          System.out.println(values[i] + " " + margins[i]);
        }
        System.out.println();
      }

      String[] checkValues = new String[]{temperature, windy, outlook, snow_cover, rainfall};

      for (int node = 0; node < bayesNet.getNrOfNodes() - 1; node++) {
        String[] outcomes = bayesNet.getValues(node);
        for (String outcome : outcomes) {
          System.out.print(outcome + " ");
        }
        System.out.println();
        int iValue = 0;
        while (iValue < outcomes.length
            && !outcomes[iValue].equals(checkValues[node])) {
          iValue++;
        }

        if (iValue == outcomes.length) {
          iValue = -1;
        }
        if (iValue < outcomes.length) {
          if (bayesNet.getEvidence(node) < 0 && iValue >= 0) {
            bayesNet.setEvidence(node, iValue);
            m_marginCalculatorWithEvidence.setEvidence(node,
                iValue);
          } else {
            bayesNet.setEvidence(node, iValue);
            SerializedObject serializedObject = new SerializedObject(m_marginCalculator);
            m_marginCalculatorWithEvidence = (MarginCalculator) serializedObject
                .getObject();
            for (int iNode = 0; iNode < bayesNet.getNrOfNodes(); iNode++) {
              if (bayesNet.getEvidence(iNode) >= 0) {
                m_marginCalculatorWithEvidence.setEvidence(iNode,
                    bayesNet.getEvidence(iNode));
              }
            }
          }
          for (int iNode = 0; iNode < bayesNet.getNrOfNodes(); iNode++) {
            bayesNet.setMargin(iNode,
                m_marginCalculatorWithEvidence.getMargin(iNode));
          }
        }
      }

      String[] values = bayesNet.getValues(bayesNet.getNrOfNodes() - 1);
      double[] margins = bayesNet.getMargin(bayesNet.getNrOfNodes() - 1);
      for (int i = 0; i < values.length; i++) {
        printWriter.println(values[i] + " " + margins[i]);
      }

      System.out.println(bayesNet);
    } catch (Exception e) {
      e.printStackTrace();
    }

    printWriter.println(temperature + " " + windy + " " + outlook + " " + snow_cover + " " + rainfall);
  }
}
