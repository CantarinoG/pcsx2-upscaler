# Material photos

The photos `material.py` takes fine detail from, one folder per family. Except where noted, they come from [Poly Haven](https://polyhaven.com/textures), published under [CC0](https://polyhaven.com/license), and were converted to 1024x1024 greyscale (only their luminance is used).

| Folder | For | Sources |
|---|---|---|
| `asphalt/` | roads | [Asphalt 01](https://polyhaven.com/a/asphalt_01), [Asphalt 04](https://polyhaven.com/a/asphalt_04), [Asphalt 05](https://polyhaven.com/a/asphalt_05) |
| `concrete/` | concrete, plaster, pavement | [Granular Concrete](https://polyhaven.com/a/granular_concrete), [Gravel Concrete](https://polyhaven.com/a/gravel_concrete), [Rough Concrete](https://polyhaven.com/a/rough_concrete) |
| `dirt/` | soil, sand, gravel | [Park Dirt](https://polyhaven.com/a/park_dirt), [Gravelly Sand](https://polyhaven.com/a/gravelly_sand), [Farm Soil](https://polyhaven.com/a/farm_soil) |
| `stone/` | rock, bark | [Rock 08](https://polyhaven.com/a/rock_08), [Rock Surface](https://polyhaven.com/a/rock_surface), [Seaside Rock](https://polyhaven.com/a/seaside_rock), [Worn Rock Natural 01](https://polyhaven.com/a/worn_rock_natural_01) |
| `masonry/` | bricks, blocks (grain only) | [Gravel Concrete](https://polyhaven.com/a/gravel_concrete), [Grey Plaster 02](https://polyhaven.com/a/grey_plaster_02), [White Plaster Rough 01](https://polyhaven.com/a/white_plaster_rough_01) |
| `grass/` | grass, moss | [Sparse Grass](https://polyhaven.com/a/sparse_grass), [Leafy Grass](https://polyhaven.com/a/leafy_grass), [Withered Grass](https://polyhaven.com/a/withered_grass) |
| `wood/` | wood grain | [Plywood](https://polyhaven.com/a/plywood), [Fine Grained Wood](https://polyhaven.com/a/fine_grained_wood), [Dark Wood](https://polyhaven.com/a/dark_wood) |
| `metal/` | sheet metal, rust | [Rust Coarse 01](https://polyhaven.com/a/rust_coarse_01), [Rusty Metal 02](https://polyhaven.com/a/rusty_metal_02), [Rusty Metal 04](https://polyhaven.com/a/rusty_metal_04) |
| `leather/` | leather, hide | [Brown Leather](https://polyhaven.com/a/brown_leather), [Leather Red 02](https://polyhaven.com/a/leather_red_02), [Leather Red 03](https://polyhaven.com/a/leather_red_03) |
| `fur/` | fur, hair | [Thatch Roof Angled](https://polyhaven.com/a/thatch_roof_angled), `generated_0` (strands drawn by a script for this project, not a photo; also CC0) |
| `cloth/` | fabric, tarps, awnings | [Hessian 230](https://polyhaven.com/a/hessian_230), [Rough Linen](https://polyhaven.com/a/rough_linen) |

Any seamlessly tiling square `.jpg` put in a folder joins that family. Pick ones without landmarks (a big crack, plank gaps, bolts, ridges): those get stamped on every texture that uses the photo. Adding a photo only changes the textures that move to it, and removing one only those that used it.
