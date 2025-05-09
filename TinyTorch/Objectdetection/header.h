#pragma once

#define OBJDECT_EXPLORE_FUNCTIONTYPE() \
        Function_GIOU

#define OBJDECT_EXPLORE_FUNCTION() \
    static Tensor giou(const Tensor& a, const Tensor& b);

#define OBJDECT_EXPLORE_funcTypeToString_() \
    FUNC_ENUM_TO_STRING(Function_GIOU)