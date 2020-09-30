#Adopted and modified from another github project


import torch
from transformers import *
import operator
from collections import OrderedDict
import sys
import traceback
import argparse
import numpy as np






def init_model(model_type,to_lower):
	logging.basicConfig(level=logging.INFO)
	
	if model_type == "bert":
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=to_lower)
		model = BertForMaskedLM.from_pretrained("bert-base-uncased")
	elif model_type == "roberta":
		tokenizer = RobertaTokenizer.from_pretrained("roberta-base",do_lower_case=to_lower)
		model = RobertaForMaskedLM.from_pretrained("roberta-base")
	elif model_type == "robbert":
		tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base",do_lower_case=to_lower)
		model = RobertaForMaskedLM.from_pretrained("pdelobelle/robbert-v2-dutch-base")
		
	model.eval()
	return model,tokenizer


def get_sentences(filename):
	sentences = []

	# open file and read the content in a list
	with open(filename, 'r') as filehandle:
		sentences = [sentence.rstrip() for sentence in filehandle.readlines()]

	return sentences
	

def get_mask(length):
	mask_index = 1
		
	return mask_index

def task(model, tokenizer, top_k, threshold, sent):
	tokenized_text = tokenizer.tokenize(sent)
	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

	# Create the segments tensors.
	segments_ids = [0] * len(tokenized_text)

	masked_index = 0

	for i in range(len(tokenized_text)):
		if (tokenized_text[i] == "entity"):
			masked_index = i
			break
	if (masked_index == 0):
		dstr = ""
		for i in range(len(tokenized_text)):
			dstr += "   " +  str(i) + ":"+tokenized_text[i]
		print(dstr)
		masked_index = get_mask(len(tokenized_text))

	tokenized_text[masked_index] = "<mask>"
	indexed_tokens[masked_index] = 50
	print(tokenized_text)
	print(masked_index)
	results_dict = {}

	# Convert inputs to PyTorch tensors
	tokens_tensor = torch.tensor([indexed_tokens])
	segments_tensors = torch.tensor([segments_ids])

	with torch.no_grad():
		predictions = model(tokens_tensor, segments_tensors)
		for i in range(len(predictions[0][0,masked_index])):
			if (float(predictions[0][0,masked_index][i].tolist()) > threshold):
				tok = tokenizer.convert_ids_to_tokens([i])[0]
				results_dict[tok] = float(predictions[0][0,masked_index][i].tolist())

	k = 0
	sorted_dict = OrderedDict(sorted(results_dict.items(), key=lambda kv: kv[1], reverse=True))
	prob_list = np.exp(list(sorted_dict.values()))/sum(np.exp(list(sorted_dict.values())))
	with open('dutch_log.txt', "a+") as filehandle:
		filehandle.writelines('\n{}\n'.format(sent)) 
		for i in sorted_dict:
			filehandle.writelines('{} {}\n'.format(i, prob_list[k])) 
			#print(i,prob_list[k])
			k += 1
			if (k > top_k):
				break
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='roberta')
    
	args = parser.parse_args()
	model,tokenizer = init_model(args.model, True)
	sent_list = get_sentences("dutch_sent_list.txt")
	for text in sent_list:
		task(model,tokenizer,20, 1, text)

	   
