# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jiaxin/yh3dr

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jiaxin/yh3dr

# Include any dependencies generated for this target.
include thirdParty/VLFeat/CMakeFiles/vlfeat.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include thirdParty/VLFeat/CMakeFiles/vlfeat.dir/compiler_depend.make

# Include the progress variables for this target.
include thirdParty/VLFeat/CMakeFiles/vlfeat.dir/progress.make

# Include the compile flags for this target's objects.
include thirdParty/VLFeat/CMakeFiles/vlfeat.dir/flags.make

# Object files for target vlfeat
vlfeat_OBJECTS =

# External object files for target vlfeat
vlfeat_EXTERNAL_OBJECTS =

thirdParty/VLFeat/libvlfeat.a: thirdParty/VLFeat/CMakeFiles/vlfeat.dir/build.make
thirdParty/VLFeat/libvlfeat.a: thirdParty/VLFeat/CMakeFiles/vlfeat.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jiaxin/yh3dr/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Linking CXX static library libvlfeat.a"
	cd /home/jiaxin/yh3dr/thirdParty/VLFeat && $(CMAKE_COMMAND) -P CMakeFiles/vlfeat.dir/cmake_clean_target.cmake
	cd /home/jiaxin/yh3dr/thirdParty/VLFeat && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vlfeat.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
thirdParty/VLFeat/CMakeFiles/vlfeat.dir/build: thirdParty/VLFeat/libvlfeat.a
.PHONY : thirdParty/VLFeat/CMakeFiles/vlfeat.dir/build

thirdParty/VLFeat/CMakeFiles/vlfeat.dir/clean:
	cd /home/jiaxin/yh3dr/thirdParty/VLFeat && $(CMAKE_COMMAND) -P CMakeFiles/vlfeat.dir/cmake_clean.cmake
.PHONY : thirdParty/VLFeat/CMakeFiles/vlfeat.dir/clean

thirdParty/VLFeat/CMakeFiles/vlfeat.dir/depend:
	cd /home/jiaxin/yh3dr && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jiaxin/yh3dr /home/jiaxin/yh3dr/thirdParty/VLFeat /home/jiaxin/yh3dr /home/jiaxin/yh3dr/thirdParty/VLFeat /home/jiaxin/yh3dr/thirdParty/VLFeat/CMakeFiles/vlfeat.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : thirdParty/VLFeat/CMakeFiles/vlfeat.dir/depend
