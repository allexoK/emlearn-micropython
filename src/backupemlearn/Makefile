# Location of top-level MicroPython directory
MPY_DIR = ../../micropython

# Architecture to build for (x86, x64, armv6m, armv7m, xtensa, xtensawin)
ARCH = x64

# The ABI version for .mpy files
MPY_ABI_VERSION := 6.1

# Location of emlearn library
EMLEARN_DIR := $(shell python3 -c "import emlearn; print(emlearn.includedir)")

DIST_DIR := ../../dist/$(ARCH)_$(MPY_ABI_VERSION)

# Name of module
# NOTE: cannot contain _ - functions do not register on the module then?
MOD = emltreeslearner

# Source files (.c or .py)
SRC = emltreeslearner.c 

# Stuff to make soft-float work
SOFTFP_O := _arm_cmpsf2.o lesf2.o _arm_fixsfsi.o fixsfsi.o eqsf2.o gesf2.o
SOFTFP_ENABLE := 0
ifeq ($(ARCH), armv6m)
	SOFTFP_ENABLE=1
endif
ifeq ($(ARCH), armv7m)
	SOFTFP_ENABLE=1
endif

ifeq ($(SOFTFP_ENABLE), 1)
	SRC_O += $(SOFTFP_O)
	CLEAN_EXTRA += $(SOFTFP_O)
endif

DIV3_XTENSAWIN_O := _divsf3.o _extendsfdf2.o _addsubdf3.o _addsubsf3.o _divdf3.o _muldf3.o _truncdfsf2.o
DIV3_XTENSAWIN_ENABLE := 0
ifeq ($(ARCH), xtensawin)
	DIV3_ENABLE=1
endif
ifeq ($(DIV3_ENABLE), 1)
	SRC_O += $(DIV3_XTENSAWIN_O)
	# CLEAN_EXTRA += $(DIV3_XTENSAWIN_O)
endif

# SRC_O += $(shell ls | grep log | grep -v lib_a-e_log.o) $(shell ls | grep cabs) $(shell ls | grep creal) $(shell ls | grep atan2) $(shell ls | grep cimag) _floatsidf.o

# SRC_O += lib_a-log.o lib_a-log2.o lib_a-logl.o lib_a-log2l.o lib_a-logbl.o lib_a-clog.o lib_a-cabs.o 
# lib_a-log.o lib_a-log2.o lib_a-logl.o lib_a-log2l.o lib_a-logbl.o
# Releases
DIST_FILE = $(DIST_DIR)/$(MOD).mpy
$(DIST_DIR):
	mkdir -p $@

$(DIST_FILE): $(MOD).mpy $(DIST_DIR)
	cp $< $@

# Include to get the rules for compiling and linking the module
include $(MPY_DIR)/py/dynruntime.mk

# CROSS is defined by the included
LIBGCC_FILENAME = $(shell $(CROSS)gcc $(CFLAGS) -print-libgcc-file-name)
$(info $(LIBGCC_FILENAME))

_arm_cmpsf2.o:
	$(CROSS)ar -x $(LIBGCC_FILENAME) $(SOFTFP_O)

libgcc:
	mkdir -p $@
	$(CROSS)ar -x $(LIBGCC_FILENAME)


CFLAGS += -I$(EMLEARN_DIR)

dist: libgcc $(DIST_FILE)

