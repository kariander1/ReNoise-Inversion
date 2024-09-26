class HuggingfaceImagesDataset(Dataset):
	
	def __init__(self, dataset_name, subset_name, data_split, source_name, target_name, opts, num_samples:int, sample_offset=0, source_transform=None, target_transform=None):
		samples_suffix = "" if num_samples == -1 else f"[{sample_offset}:{sample_offset+num_samples}]"
		self.dataset = load_dataset(dataset_name, subset_name, split=f"{data_split}"+samples_suffix)
		# self.dataset = load_dataset(dataset_name)[self.split]
		self.split = data_split
		self.source_name = source_name
		self.target_name = target_name
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		sample = self.dataset[index]
		from_im = sample[self.source_name]
		to_im = sample[self.target_name]
		
		if isinstance(from_im, dict):
			from_im = Image.open(BytesIO(from_im['bytes']))
		if isinstance(to_im, dict):
			to_im = Image.open(BytesIO(to_im['bytes']))
            
		if self.source_transform and not self.target_transform:
      		# Patch for PairedTransform
			from_im, to_im = self.source_transform(from_im, to_im)
		else:
			if self.source_transform:
				from_im = self.source_transform(from_im)
			if self.target_transform:
				to_im = self.target_transform(to_im)
   
		samples = {
			"source": from_im,
			"target": to_im,
			"prompt": sample['prompt']
		}
		return samples

	def get_paths(self):
		paths = []
		for sample in self.dataset:
			# We assume that the prompt is a unique identifier for each image
			paths.append(sample['prompt']) 
		return paths

	def shuffle(self):
		self.dataset.shuffle()
  
  
  class PieBenchmarkDataset(HuggingfaceImagesDataset):
	def __init__(self, dataset_name, subset_name, data_split, source_name, target_name, opts, num_samples:int, sample_offset=0, source_transform=None, target_transform=None):
		super().__init__(dataset_name, subset_name, data_split, source_name, target_name, opts, num_samples, sample_offset, source_transform, target_transform)
     
	def __getitem__(self, index):
		sample = self.dataset[index]
		from_im = sample[self.source_name]
		to_im = sample[self.target_name]
		
		if isinstance(from_im, dict):
			from_im = Image.open(BytesIO(from_im['bytes']))
		if isinstance(to_im, dict):
			to_im = Image.open(BytesIO(to_im['bytes']))
            
		if self.source_transform:
			from_im = self.source_transform(from_im)
		if self.target_transform:
			to_im = self.target_transform(to_im)

		samples = {
			"source": from_im,
   			"target": to_im,
			"source_prompt": sample['source_prompt'],
   			"target_prompt": sample['target_prompt'],
      		"edit_action": sample['edit_action'],
		}
		return samples
