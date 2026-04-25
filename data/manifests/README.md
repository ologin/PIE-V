# Manifests

`take_names.txt` lists the 50 Ego-Exo4D takes used by the PIE-V paper.

The corresponding Ego-Exo4D annotations and videos are local inputs. The default
layout is defined in `configs/defaults.toml`:

```text
local/egoexo4d/
  split_50.json
  keystep_train.json
  videos_ego/
    <take_name>.mp4
```
