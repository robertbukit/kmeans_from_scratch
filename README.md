# Review
These file were generate for implemented what the kmeans algorithm works.
The kmeans_scratch.py file is main structure for build kmeans, the flow code is follow that statistical kmeans natively. which is as follows:
- initiate random centroid from the data (including thereshold and default max iteration)
- assign each data to the each centroid (assign with euclidian distance and take the minimum data value from closest cluster)
- update centroid with calculate mean of each cluster
- loop the 2nd and 3rd step to reach convergent value.

# Output
The output file take from test_kmeans.py's file. which is the output is:
Testing K = 2
Converged after 2 iterations
Cluster size: [150, 150]

Testing K = 3
Converged after 7 iterations
Cluster size: [75, 150, 75]

Testing K = 4
Converged after 5 iterations
Cluster size: [75, 74, 76, 75]

Testing K = 5
Converged after 8 iterations
Cluster size: [37, 38, 74, 76, 75]


<img width="1366" height="663" alt="Figure_1" src="https://github.com/user-attachments/assets/be58cef7-a45d-4d43-a59d-42283ad36bd0" />
