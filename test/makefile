OBJDIR := obj
DEPDIR := $(OBJDIR)/.deps
SRCDIR := src

INCS=-I ../main/include -I include -I ~/.local/include
CC=g++
CPPFLAGS=-std=c++14
DEBUG=-g
#LIBS = -lokutil
#LIB_PATHS = -L ~/Documents/prj/okutil
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d

SRCS := $(shell find src -name \*.cpp)
OBJS := $(SRCS:src/%.cpp=obj/%.o)

main: $(OBJS)
	$(CC) $(CPPFLAGS) $^ -o$@ $(INCS) $(LIB_PATHS) $(LIBS) 

test: $(TEST_OBJS)
	$(CC) $(CPPFLAGS) $^ -o$@ $(INCS) $(LIB_PATHS) $(LIBS) 

#$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPDIR)/%.d | $(DEPDIR)
	$(CC) $(DEBUG) $(DEPFLAGS) $(CPPFLAGS) -c $< -o$@ $(INCS)

$(DEPDIR): ; @mkdir -p $@

DEPFILES := $(SRCS:$(SRCDIR)/%.cpp=$(DEPDIR)/%.d)
$(DEPFILES):

include $(wildcard $(DEPFILES))

