# Knee osteoarthritis

Osteoarthritis (OA) is the most common joint disease in the world. Its risk factors include obesity, joint injuries, age, and repeated stress on the joint. Knee OA in its advanced stages causes severe pain and often leads to knee joint replacement surgery. A useful way to avoid or at least postpone such medical interventions is to detect OA in its early stages.

Grading systems used to assess the severity of OA from X-ray images include Osteoarthritis Research Society International (OARSI) and Kellgren–Lawrence (KL). KL classification consists of grades 0 to 4, 0 meaning no OA, 1 doubtful, 2 minimal, 3 moderate, and 4 severe OA. Assigning a KL grading to a radiograph is basically a multiclass classification problem, which gives a reason to assume that convolutional neural networks can be trained to solve the problem. Commonly used datasets are the data from Osteoarthritis Initiative (OAI) and Multicenter Osteoarthritis Study (MOST).

Overall, it seems that the problem of automatic KL grading will not be solved by applying different neural network architectures. Therefore, other means, such as inclusion of prior knowledge in the model development or enriching the diversity and quality of data will be needed.

We have developed two methods that utilize such prior knowledge.

* Sharpening of intercondylar tubercles, or tibial spiking, has been hypothesized to be an early sign of knee OA. Our [multi-scale network](tibial-spiking) pays attention to the intercondylar tubercles as well as the knee joint in general.
* Not all pixels in X-ray image are equally important. We trained neural networks with only [lateral and medial edges](edge-parts).
