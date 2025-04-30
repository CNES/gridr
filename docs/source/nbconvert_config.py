c = get_config()

c.TagRemovePreprocessor.enabled = True
c.TagRemovePreprocessor.remove_cell_tags = {"remove_cell"}
c.TagRemovePreprocessor.remove_input_tags = {"remove_input"}

c.Exporter.preprocessors = ["nbconvert.preprocessors.TagRemovePreprocessor"]

