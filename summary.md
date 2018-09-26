# Summary 

This is a quick summary of what I've worked on so far. I have collected and read through a few of the following papers, in addition to the initial readings that were sent in an email over the summer.

* CryptoNets: Applying Neural Networks to Encrypted Data with High Throughput and Accuracy [February 24, 2016]
* Somewhat Practical Fully Homomorphic Encryption [Junfeng Fan and Frederik Vercauteren]
* Fully Homomorphic Encryption without Bootstrapping [Zvika Brakerski, Craig Gentry, and Vinod Vaikuntanathan]
* Fully Homomorphic Encryption without Modulus Switching from Classical GapSVP [Zvika Brakerski]

I've also been using the [Homomorphic Encryption Standardization](http://homomorphicencryption.org/) website to learn more about best practises and try to get a feel for what people think are the best homomorphic algorithms to use. It seems that FV/BFV or BGV seem to be recommended? Correct me if I'm wrong.

In addition to this I've had a look at the SEAL library and the ipython notebook you sent me demonstrating how to use it. It looks great but I'm not yet sure how to make this work with deep learning libraries though or if that's even possible....

I think the main thing I'd like to clarify in the meeting is what direction to go for the project. I think possibly one of the following ideas:

* Try out a few different HE algorithms and assess speed and accuracy with a particular network structure? However I'm not sure if there are many python libraries like SEAL that easily implement these. And if not how I would go about this in general...
* Try one of the HE algorithms that's regarded as one people should use (current recommended standard from the above website) and implement it in a variety of different network structures and assess speed/accurary etc...
* Some combination of both of these
