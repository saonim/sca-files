import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.RandomAccessFile;
import java.util.Arrays;

public class CSVAverage {
    public static void main(String[] args) throws Exception {
	String input = "/home/saoni/exp_results/empty_kernel.csv";
	String output = "./temp.bin";
	String outputFileName = "./empty_kernel_average.csv";
	int col=Integer.MAX_VALUE;
	int row=0;
	BufferedReader br = new BufferedReader(new FileReader(input));
	FileOutputStream fos = new FileOutputStream(output);
	String line;
	int i=0;
	while ((line = br.readLine()) != null) {
	    line = line.replace("[", "");
	    line = line.replace("]", "");
	    line = line.replace("|", "");
	    String[] numbers = line.split(",");
	    col=Math.min(col,numbers.length);
	    i++;
	    System.out.println("Loop1 " + i);
	}
	br.close();
	br = new BufferedReader(new FileReader(input));

	try {
	    while ((line = br.readLine()) != null) {
		row++;
		line = line.replace("[", "");
		line = line.replace("]", "");
		line = line.replace("|", "");
		String[] numbers = line.split(",");
		numbers=Arrays.copyOfRange(numbers, 0, col);          
		for (String number : numbers) {
		    fos.write(Integer.parseInt(number.trim()));
		}
		System.out.println("Loop2 " + row);
	    }
	} finally {
	    br.close();
	    fos.close();

	}
        RandomAccessFile randomAccessFile= new RandomAccessFile(output, "r");
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFileName));

	try {
	    for(int c=0;c<col;++c){
		long sum=0;
		for(int r=0;r<row;++r){
		    randomAccessFile.seek((long)r*col+c);
		    sum+=randomAccessFile.read();
		}
		writer.append((sum/(double)row)+"\n");
		System.out.println("Loop3 " + c);
	    }
	} finally {
	    randomAccessFile.close();
	    writer.close();

	}
    }
}
