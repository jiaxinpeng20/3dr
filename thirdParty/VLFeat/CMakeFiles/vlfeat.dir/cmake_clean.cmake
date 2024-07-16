file(REMOVE_RECURSE
  "libvlfeat.a"
  "libvlfeat.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/vlfeat.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
