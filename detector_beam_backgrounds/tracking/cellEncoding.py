
metadata = reader.get("metadata")[0]
cellid_encoding = metadata.get_parameter("CDCHHits__CellIDEncoding")
decoder = dd4hep.BitFieldCoder(cellid_encoding)
superLayer = decoder.get(cellID, "superLayer")
layer = decoder.get(cellID, "layer")
phi = decoder.get(cellID, "phi")
stereo = decoder.get(cellID, "stereo")