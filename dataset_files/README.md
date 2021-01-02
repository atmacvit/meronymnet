Pre processing script:

Use the Download and Preprocessing script to convert the .mat annotation files into json files.

The json file has the following format:

For each <img>.mat file, and annotation dictionary is created whose keys are the object annotations. For image with object person, this dictionary would have following keys: ['person']. At each key, another dictionary is score which has the mask, bbox and parts information. So for the above example dict['person'].keys() would return ['mask','bbox','parts']. The mask and bbox arrays have been converted to lists for serialization. Parts key holds part annotation dictionary where the keys are part names. Each part name has dictionary with ['mask','bbox'] as keys. These keys hold the mask and bbox matrix as list.
