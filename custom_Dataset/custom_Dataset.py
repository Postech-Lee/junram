"""custom_Dataset dataset."""

import tensorflow_datasets as tfds

# TODO(custom_Dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(custom_Dataset): BibTeX citation
_CITATION = """
"""


class CustomDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for custom_Dataset dataset."""
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
     data.zip files should be located at /root/tensorflow_dataset/downloads/manual
     """
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
      """Returns the dataset metadata."""
      # TODO(custom_dataset): Specifies the tfds.core.DatasetInfo object
      return tfds.core.DatasetInfo(
          builder=self,
          description=_DESCRIPTION,
          features=tfds.features.FeaturesDict({
              # These are the features of your dataset like images, labels ...
              'image': tfds.features.Image(shape=(None, None, 3)),
              'label': tfds.features.ClassLabel(names=['no', 'yes']),
              'custom_text': tfds.features.Text()
          }),
          # If there's a common (input, target) tuple from the
          # features, specify them here. They'll be used if
          # `as_supervised=True` in `builder.as_dataset`.
          supervised_keys=('image', 'label'),  # Set to `None` to disable
          homepage='https://dataset-homepage/',
          citation=_CITATION,
      )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(custom_dataset): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')
    archive_path = dl_manager.manual_dir / 'data.zip'
    extracted_path = dl_manager.extract(archive_path)

    # TODO(custom_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path = extracted_path/'train'),
        'test': self._generate_examples(path = extracted_path/'test'),
    }

  def _generate_examples(self, images_path, label_path):
    # Read the input data out of the source files
    with label_path.open() as f:
      for row in csv.DictReader(f):
        image_id = row['image_id']
        # And yield (key, feature_dict)
        yield image_id, {
            'image_description': row['description'],
            'image': images_path / f'{image_id}.jpeg',
            'label': row['label'],
        }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(custom_dataset): Yields (key, example) tuples from the dataset
    for f in path.glob('*.jpeg'):
      yield 'key', {
          'image': f,
          'label': 'yes',
      }