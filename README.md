# bert_bias
This is a project to investigate bias in contextual language models and trying to mitigate it.

We start off with a masked language modelling task. bert_mlm.py generates a ranked list for the probability of a given word being the masked word in a given sentence.

We used the following sentences as an example: 
the teacher was a <mask>
  
The top 10 possible outputs for which were: 
woman 0.05570428366662058
man 0.04102633483610361
friend 0.04059647814295565
student 0.03954510512991802
child 0.026983342923277063
story 0.02219460654955375
future 0.014575299636183507
girl 0.013476121394806546
winner 0.013310269401394724
band 0.01131433537306671
song 0.010252867396018292
