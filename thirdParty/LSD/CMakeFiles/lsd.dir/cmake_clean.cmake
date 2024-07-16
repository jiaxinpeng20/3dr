file(REMOVE_RECURSE
  "liblsd.a"
  "liblsd.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/lsd.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
