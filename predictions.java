//Vinay Suri
//neural network predictor 

import java.io.*;
import java.util.Random;
import java.lang.Math;
import java.lang.Integer;
import java.util.Collections;
import java.util.ArrayList;

class faceRecognizer{
	int numHiddenLayers;
	int numInputs;
	int numOutputs;
	public int seed = 1;
	double[] inputs;
	double SQUASH = 100;
	double eta = 0.3;
	double momentum = 0.3;
	double[] hiddens;
	double[] outputs;
	double[][] inputWeights;
	double[][] hiddenWeights;
	double[] hiddenDeltas;
	double[] outputDeltas;
	double[][] inputPrevWeights;
	double[][] hiddenPrevWeights;
	int current;

	public faceRecognizer()
	{
		numHiddenLayers = 3 + 1;
		numInputs = 128 * 120 + 1;
		numOutputs = 1 + 1;
		inputs = new double[numInputs];
		hiddens = new double[numHiddenLayers];
		outputs = new double[numOutputs];
		inputWeights = new double[numHiddenLayers][numInputs];
		hiddenWeights = new double[numOutputs][numHiddenLayers];
		hiddenDeltas = new double[numHiddenLayers];
		outputDeltas = new double[numOutputs];
		inputPrevWeights = new double[numHiddenLayers][numInputs];
		hiddenPrevWeights = new double[numOutputs][numHiddenLayers];
		zeroOutArrays();
                randomizeWeights();
		current = 1;
	}

	public void zeroOutArrays()
	{
		for(int i = 0; i < numHiddenLayers; i++)
		{
			hiddens[i] = 0;
			hiddenDeltas[i] = 0;
		}
		for(int i = 0; i < numOutputs; i++)
		{
			outputs[i] = 0;
			outputDeltas[i] = 0;
		}
		for(int i = 0; i < numHiddenLayers; i++)
			for(int j = 0; j < numInputs; j++){
                                inputWeights[i][j] = 0;
				inputPrevWeights[i][j] = 0;
                        }

		for(int i = 0; i < numOutputs; i++)
			for(int j = 0; j < numHiddenLayers; j++){
                                hiddenWeights[i][j] = 0;
				hiddenPrevWeights[i][j] = 0;
                        }
	}

	public double sigmoid(double x)
	{
		return (1.0 / (1 + Math.exp(-x)));
	}

	// gets all input nodes of network
	public void getInput(String filename)
	{
		current = 1;
            try{
        	FileInputStream fstream = new FileInputStream(filename);
        	// Get the object of DataInputStream
        	DataInputStream in = new DataInputStream(fstream);
        	BufferedReader br = new BufferedReader(new InputStreamReader(in));
        	String strLine;

        	while((strLine = br.readLine()) != null)
			{				
        		String[] tokens = strLine.split(" ");
				for(int i = 0; i < tokens.length; i++)
				{
					inputs[current] = Integer.parseInt(tokens[i])/255.0;
					current = current + 1;
				}
			}
			// tests if input is correct
			/*for(int i = 0; i < current; i++)
				System.out.print(inputs[i] + " ");*/
			in.close();
		}
		catch (Exception e){//Catch exception if any
			e.printStackTrace();
			System.err.println("Error: " + e.getMessage());
		}
        
	}


	// Randomizes weights on all edges
	public void randomizeWeights()
	{
		Random random = new Random(seed);
		for(int i = 0; i < numHiddenLayers; i++)
			for(int j = 0; j < numInputs; j++)
				inputWeights[i][j] = ((random.nextDouble() * 2) - 1.0) / SQUASH;

		for(int i = 0; i < numOutputs; i++)
			for(int j = 0; j < numHiddenLayers; j++)
				hiddenWeights[i][j] = ((random.nextDouble() * 2) - 1.0) / SQUASH;

		// test if randomizing working
		/*for(int i = 0; i < numOutputs; i++)
			for(int j = 0; j < numHiddenLayers; j++)
				System.out.print(hiddenWeights[i][j] + " ");*/
		
	}

	// computes outputs when the input is fed throught the network. Returns final output
	public double feedForward()
	{
		inputs[0] = 1; // constant term

		for(int i = 1; i < numHiddenLayers; i++)
		{
			double sum = 0;
			for(int j = 0; j < numInputs; j++)
			{
				sum = sum + inputs[j]*inputWeights[i][j];
			}
			hiddens[i] = sigmoid(sum);
		}


		hiddens[0] = 1;
		for(int i = 1; i < numOutputs; i++)
		{
			double sum = 0;
			for(int j = 0; j < numHiddenLayers; j++)
			{
				sum = sum + hiddens[j]*hiddenWeights[i][j];
			}
			outputs[i] = sigmoid(sum);
		}

		return outputs[1];
	}

	// find errors in network based on whether output = answer
	public void calculateErrors(int answer)
	{
		outputDeltas[1] = outputs[1]*(1.0-outputs[1])*(answer - outputs[1]);

		for(int i = 1; i < numHiddenLayers; i++)
		{
			hiddenDeltas[i] = hiddens[i]*(1.0-hiddens[i])*hiddenWeights[1][i]*outputDeltas[1];
		}

	}

	// update weights in graph based on error
	public void updateWeights()
	{
		hiddens[0] = 1.0;
		for(int i = 1; i < numOutputs; i++)
		{
			for(int j = 0; j < numHiddenLayers; j++)
			{
				double newPrevWeight = outputDeltas[i]*eta*hiddens[j] + momentum*hiddenPrevWeights[i][j];
				hiddenWeights[i][j] = hiddenWeights[i][j] + newPrevWeight;
				hiddenPrevWeights[i][j] = newPrevWeight;
			}
		}

		inputs[0] = 1;
		for(int i = 1; i < numHiddenLayers; i++)
		{
			for(int j = 0; j < numInputs; j++)
			{
				double newPrevWeight = hiddenDeltas[i]*eta*inputs[j] + momentum*inputPrevWeights[i][j];
				inputWeights[i][j] = inputWeights[i][j] + newPrevWeight;
				inputPrevWeights[i][j] = newPrevWeight;
			}
		}
	}


	public void train(String[] Female, String[] Male)
	{
            int Epoch = 100;
            for(int currentEpoch = 0; currentEpoch < Epoch; currentEpoch++)
            {
                //System.out.println(currentEpoch+1 + "% Trained");
		int femaleCounter = 0;
		int maleCounter = 0;
		boolean m = true;
		while(true)
		{
			if(m == true && maleCounter < Male.length)
			{
				if(Male[maleCounter] != null && Male[maleCounter].endsWith(".txt"))
				{
					//System.out.println(maleNames[maleCounter]);
					getInput("./Male/" + Male[maleCounter]);
					double output = feedForward();
					int answer = 1;
					calculateErrors(answer);
					updateWeights();
                                               /*
					if(currentEpoch == Epoch -1)
					{
						System.out.println("Male: 1 Output: " + output);
						//saveFile("./Male2/" + maleNames[maleCounter]);
					}
                                               */
				}
					maleCounter += 1;
			}
				
			if(m == false && femaleCounter < Female.length)
			{
				if(Female[femaleCounter] != null && Female[femaleCounter].endsWith(".txt"))
				{
					//System.out.println(femaleNames[femaleCounter]);
					getInput("./Female/" + Female[femaleCounter]);
					double output = feedForward();
					int answer = 0;
					calculateErrors(answer);
					updateWeights();
                                               /*
					if(currentEpoch == Epoch -1)
					{
						System.out.println("Female: 0 Output: " + output);
						//saveFile("./Female2/" +  femaleNames[femaleCounter]);
					}
                                               */
				}
				femaleCounter += 1;
			}
			m = !m;
			if(maleCounter >= Male.length || femaleCounter >= Female.length)
				break;
			}
			m = !m;
			}

		
		
	}

        public int[] crossvalidate(String[] Female, String[] Male)
	{
            int Epoch = 1;

            int[] outputs = new int[Female.length+Male.length];
            for(int currentEpoch = 0; currentEpoch < Epoch; currentEpoch++)
            {
		int femaleCounter = 0;
		int maleCounter = 0;
                int outputCounter = 0;
		boolean m = true;
		while(true)
		{
			if(m == true && maleCounter < Male.length)
			{
				if(Male[maleCounter] != null && Male[maleCounter].endsWith(".txt"))
				{
					//System.out.println("Male File: " + Male[maleCounter] + " maleCounter: " + maleCounter);
					getInput("./Male/" + Male[maleCounter]);

					double output = feedForward();
                                        if(output > .5)
                                            outputs[outputCounter] = 1;
                                        else
                                            outputs[outputCounter] = 0;
                                        outputCounter++;
					//int answer = 1;
					//calculateErrors(answer);
					//updateWeights();
				}
					maleCounter += 1;
			}

			if(m == false && femaleCounter < Female.length)
			{
				if(Female[femaleCounter] != null && Female[femaleCounter].endsWith(".txt"))
				{
					//System.out.println("Female File: " + Female[femaleCounter] + " femaleCounter: " + femaleCounter);
					getInput("./Female/" + Female[femaleCounter]);
					double output = feedForward();
                                        if(output < .5)
                                            outputs[outputCounter] = 1;
                                        else
                                            outputs[outputCounter] = 0;
                                        outputCounter++;
					//int answer = 0;
					//calculateErrors(answer);
					//updateWeights();
				}
				femaleCounter += 1;
			}
			m = !m;
			if(maleCounter >= Male.length || femaleCounter >= Female.length)
				break;
		}
		m = !m;
            }
            return outputs;


	}

	public int[] test(String[] Test)
	{
                        int[] outputs = new int[Test.length];
			int Epoch = 1;
			for(int currentEpoch = 0; currentEpoch < Epoch; currentEpoch++)
			{
                            int outputCounter = 0;
                            int testCounter = 0;
                            while(true)
                            {
				if(testCounter < Test.length)
				{
					if(Test[testCounter].endsWith(".txt"))
					{
                                                int answer;
						//System.out.println(Test[testCounter]);
						getInput("./Test/" + Test[testCounter]);
						double output = feedForward();

						String name = new String();
						int confidence = 0;
                                                if(output > .5){
							confidence = (int)((output)*100);
                                                	name = "Male";
                                                }
                                                else{
							confidence = (int)((1-output)*100);
                                                	name = "Female";
                         			}
						System.out.println(name + ": " + confidence + "%");
                                                
					}
					testCounter += 1;
				}
                                outputCounter++;
				if(testCounter >= Test.length)
					break;
                            }
			}
                        return outputs;
	}

        public double validate(String[] femaleArray, String[] maleArray)
        {
                int folds = 5;

                int femalesectionsize = femaleArray.length/folds;

                int malesectionsize = maleArray.length/folds;
                
                //System.out.println("Female Length: " + femaleArray.length + " Sectoion Size: " + femalesectionsize);
                //System.out.println("Male Length: " + maleArray.length + " Section Size: " + malesectionsize);

		ArrayList<String> femaleNames = new ArrayList<String>();
		ArrayList<String> maleNames = new ArrayList<String>();

		for(int i = 0; i < femaleArray.length; i++)
                {
                    //System.out.println("Adding " + femaleArray[i]);
                    femaleNames.add(femaleArray[i]);
                }

                //System.out.println("Female Array Initialized");

		for(int i = 0; i < maleArray.length; i++)
                {
                    //System.out.println("Adding " + maleArray[i]);
                    maleNames.add(maleArray[i]);
                }

                //System.out.println("Male Array Initialized");

		int testListLength = Math.min(femaleArray.length, maleArray.length);
		int foldLength = testListLength/folds;
		String[][] femaleSamples = new String[folds][foldLength];
		String[][] maleSamples = new String[folds][foldLength];

                //System.out.println("Sample Arrays Initialized");

		Random random = new Random(seed);
		for(int i = 0; i < folds; i++)
                    for(int j = 0; j < foldLength; j++)
                    {
                        femaleSamples[i][j] = femaleNames.remove(Math.abs(random.nextInt()%femaleNames.size()));
                    }
                //System.out.println("Random Female Samples Produced");
                for(int i = 0; i < folds; i++)
                    for(int j = 0; j < foldLength; j++)
                    {
                        maleSamples[i][j] = maleNames.remove(Math.abs(random.nextInt()%maleNames.size()));
                    }
                 
                //System.out.println("Random Male Samples Produced");

                
                //double[] SuccessRates = new double[folds];
                double TotalSuccess = 0;
                
                for(int i = 0; i < folds; i++){
                    for(int j = 0; j < folds-1; j++){
                        System.out.println("Training Against Fold: " + (i+j)%folds);
                        train(femaleSamples[(i+j)%folds], maleSamples[(i+j)%folds]);
                    }
                    int[] outputs;
                    System.out.println("Testing Against Fold: " + (i+folds-1)%folds);
                    outputs = crossvalidate(femaleSamples[(i+folds-1)%folds], maleSamples[(i+folds-1)%folds]);
                    //SuccessRates[i] = stats(outputs);
                    TotalSuccess +=  stats(outputs);
                    zeroOutArrays();
                    randomizeWeights();

                }
                TotalSuccess /= folds;
                System.out.println("Total Average Success: " + TotalSuccess);
                return TotalSuccess;
        }

        public double stats(int[] outputs)
        {
                double TotalSuccess = 0;
                double MeanSuccess;
                double SDSuccess = 0;
                double TotalError = 0;
                double MeanError;
                double SDError = 0;

                for(int i = 0; i < outputs.length; i++)
                {
                    TotalSuccess += outputs[i];
                }

                //TotalError = outputs.length - TotalSuccess;
                MeanSuccess = TotalSuccess/(outputs.length);
                MeanError = TotalError/(outputs.length);

                for(int i = 0; i < outputs.length; i++){
                    SDSuccess += Math.pow((outputs[i] - MeanSuccess),2);
                    //SDError += Math.pow(((1- outputs[i]) - MeanError),2);
                }

                //SDSuccess = Math.sqrt(SDSuccess/(outputs.length));
                System.out.println("Test Complete");
                System.out.println("Success Rate: " + MeanSuccess);
                //System.out.println("Standard Deviation of Success: " + SDSuccess);
                //System.out.println("Mean of Error: " + MeanError);

                return MeanSuccess;
        }

	public void saveFile(String filename)
	{
		try{
		FileWriter fo = new FileWriter(filename);
		BufferedWriter out = new BufferedWriter(fo);
		for(int i = 1; i < current; i++)
		{
			out.write(inputs[i] + " ");
			if(i % 16 == 0)
				out.write('\n');
		}
		out.close();
		}
		catch(Exception e)
		{
			e.printStackTrace();
			System.err.println("Bad Write");
		}
	}
};

class Trainer{

    faceRecognizer face;

	public Trainer(){
		face = new faceRecognizer();
	}
        public void train(String Female, String Male){
                File female = new File(Female);
                File male = new File(Male);
		String[] femalelist = female.list();
                String[] malelist = male.list();
            	face.train(femalelist, malelist);
        }
        public void test(String Test){
                File test = new File(Test);
                String[] testlist = test.list();
            	face.test(testlist);
        }
        public void validate(String Female, String Male){
            	File female = new File(Female);
                File male = new File(Male);
		String[] femalelist = female.list();
                String[] malelist = male.list();

                double[] SuccessRates = new double[10];


                for(int i = 0; i < 10; i++){
                    System.out.println("Test: " + (i+1));
                    int oldseed = face.seed;
                    SuccessRates[i] = face.validate(femalelist, malelist);
                    face = new faceRecognizer();
                    face.seed = oldseed++;
                }

                double SDSuccess = 0;
                double MeanSuccess = 0;

                for(int i = 0; i < SuccessRates.length; i++)
                    MeanSuccess += SuccessRates[i];
                MeanSuccess /= 10;

                for(int i = 0; i < SuccessRates.length; i++)
                    SDSuccess += Math.pow((SuccessRates[i] - MeanSuccess),2);
                SDSuccess = Math.sqrt(SDSuccess);
                System.out.println("All Ten Iterations of Cross Validation Complete");
                System.out.println("Success Mean: " + MeanSuccess);
                System.out.println("Standard Deviation: " + SDSuccess);
        }
        public faceRecognizer getFace(){
            return face;
        }
};

public class predictions{


	public static void main(String args[])
	{
                /*
                1. Fully describe the network architecture and why it was chosen. How many input/hidden/output layer nodes?
                Fully connected or partially connected? etc. Submit your code as: <teamname.java>. Your code should be
                compilable on our CSIF machines and have a -train option (that looks in the female (here) and male directory
                (here) and -test option that reads in the data (from the here directory) in the format I have given it to you. (20
                points
                */

                // Parse through the command line arguements
                String[] filenames = new String[3];
                filenames[0] = "./Female";
                filenames[1] = "./Male";
                filenames[2] = "./Test";
                Trainer trainer = new Trainer();
		try
		{
                    if(args[0].equalsIgnoreCase("-train"))
                        trainer.train(filenames[0], filenames[1]);
                    else if(args[0].equalsIgnoreCase("-test")){
                        trainer.train(filenames[0], filenames[1]);
                        trainer.test(filenames[2]);
                    }
                    else if(args[0].equalsIgnoreCase("-validate"))
                        trainer.validate(filenames[0], filenames[1]);
                    else
                        throw new IllegalArgumentException();
		}
		catch(IllegalArgumentException ia)
		{
			System.err.println("Invalid Arguments: " + ia.getMessage());
                        System.err.println("Use -train Filename to train with data");
                        System.err.println("Use -test Filename to test with data");
                        System.err.println("Use -validate Filename to validate with data");
			System.exit(4);
		}

	}
};

