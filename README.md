## Machine Learn examples 

This is a sample package to preform qml and ml prediction models.

### Docker

It is recommended to use docker so as to leave the host system unmodified. To build Docker container run:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   docker build -t qml .

To connect to and run docker container:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  docker run -it qml

Once in the container the program can be run by:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   python3 run.py

nano is installed in the container and can be used to view the files.


### Native installation

Once the repository has been cloned, navigate to qml. In the directory run:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; pip3 install -r requirements.txt

Then you can run the program as stated above.


### File discription

*classical_analysis.py* has function for creating a classical ai prediction model.

*quantum_analysis.py* have function for creating a qnn and vqc prediction model.

*main.py* combines the method in the previous two files into simple function.

*plotting.py* has functions for post-processing of the results, printing results to file, reading results from file, and for plotting graphs.

*run.py* is the main program. The tolerance for classical and qnn can be set using \-\-tol=value. tol is set to 0.02 by default. The vqc can be turn on by using \-\-vqc=True. vqc is turned off by default.

