// Include MicroPython API.
#include "py/dynruntime.h"

#define DTR_DEBUG_PRINT(X) mp_printf(&mp_plat_print,"%s",X);
#define DTR_DEBUG_PRINTLN(X) mp_printf(&mp_plat_print,"%s\n\r",X);
#define DTR_DEBUG_PRINT_F(X) mp_printf(&mp_plat_print,"%f",(double)X);
#define DTR_DEBUG_PRINT_U(X) mp_printf(&mp_plat_print,"%lu",(uint32_t)X);
#define tdc_malloc(num_of_bytes) m_malloc(num_of_bytes)
#define tdc_realloc(ptr,old_num_bytes,new_num_bytes) m_realloc(ptr,new_num_bytes)
#define tdc_free(ptr,num_of_bytes) m_free(ptr)

#include "/home/alex/Documents/PlatformIO/Projects/decisionTreeC/lib/TinyDecisionTreeClassifierC/src/TinyDecisionTreeClassifierC/TinyDecisionTreeClassifierC.h"
// #include "/home/alex/Documents/PlatformIO/Projects/decisionTreeC/lib/TinyDecisionTreeClassifierC/src/TinyDecisonTreeClassifierLoaderC/TinyDecisionTreeClassifierLoaderC.h"

// memset is used by some standard C constructs
#if !defined(__linux__)
void *memcpy(void *dst, const void *src, size_t n) {
    return mp_fun_table.memmove_(dst, src, n);
}
void *memset(void *s, int c, size_t n) {
    return mp_fun_table.memset_(s, c, n);
}
#endif

// This structure represents Eml Tree Learner instance objects.
typedef struct _emltreeslearner_EmlTreeLearner_obj_t {
    mp_obj_base_t base;
    mp_uint_t max_depth;
    mp_uint_t min_sample_split;
    struct Node* root;
} emltreeslearner_EmlTreeLearner_obj_t;

mp_obj_full_type_t emltreeslearner_type_EmlTreeLearner;

// This represents EmlTreeLearner.__new__ and EmlTreeLearner.__init__, which is called when
// the user instantiates a EmlTreeLearner object.
STATIC mp_obj_t emltreeslearner_EmlTreeLearner_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    // Allocates the new object and sets the type.
    mp_arg_check_num(n_args, n_kw, 3, 3, false);
    
    emltreeslearner_EmlTreeLearner_obj_t *self = mp_obj_malloc(emltreeslearner_EmlTreeLearner_obj_t, type);
    mp_uint_t trees_number =  mp_obj_get_int(args[0]);
    if(trees_number!=1){
        mp_raise_ValueError(MP_ERROR_TEXT("For now only one tree is supported"));
    }
    self->max_depth = mp_obj_get_int(args[1]);
    self->min_sample_split = mp_obj_get_int(args[2]);
    self->root = NULL;
    return MP_OBJ_FROM_PTR(self);
}

STATIC mp_obj_t emltreeslearner_EmlTreeLearner_fit(size_t n_args, const mp_obj_t *args) {
    mp_obj_t self_in = args[0];
    mp_obj_t X = args[1];
    mp_obj_t Y = args[2];
    mp_obj_t row_stride = args[3];

    emltreeslearner_EmlTreeLearner_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if(!mp_obj_is_int(row_stride)){
        mp_raise_ValueError(MP_ERROR_TEXT("Expecting int array on row_stride"));
    }
    uint32_t row_str = mp_obj_get_int(row_stride);

    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(X, &bufinfo, MP_BUFFER_RW);
    if (bufinfo.typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("Expecting float array on Y"));
    }

    mp_buffer_info_t bufinfo0;
    mp_get_buffer_raise(Y, &bufinfo0, MP_BUFFER_RW);
    if (bufinfo0.typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("Expecting float array on Y"));
    }

    if((mp_uint_t)((bufinfo.len)/sizeof(float)/row_str)!=(mp_uint_t)((bufinfo0.len)/sizeof(float))){
        mp_raise_ValueError(MP_ERROR_TEXT("X and Y have different number of rows"));    
    }

    tdc_fit(&self->root,(float*)bufinfo.buf,(float*)bufinfo0.buf,(bufinfo0.len)/sizeof(float),row_str,self->max_depth,self->min_sample_split);
    return mp_const_none;
}
STATIC MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(emltreeslearner_EmlTreeLearner_fit_obj, 4, 4, emltreeslearner_EmlTreeLearner_fit);

STATIC mp_obj_t emltreeslearner_EmlTreeLearner_plot(mp_obj_t self_in) {
    emltreeslearner_EmlTreeLearner_obj_t *self = MP_OBJ_TO_PTR(self_in);
    tdc_plot(self->root,0);
    return mp_const_none;
}
STATIC MP_DEFINE_CONST_FUN_OBJ_1(emltreeslearner_EmlTreeLearner_plot_obj, emltreeslearner_EmlTreeLearner_plot);

STATIC mp_obj_t emltreeslearner_EmlTreeLearner_predict(mp_obj_t self_in, mp_obj_t X) {
    emltreeslearner_EmlTreeLearner_obj_t *self = MP_OBJ_TO_PTR(self_in);
    mp_buffer_info_t bufinfo;

    mp_get_buffer_raise(X, &bufinfo, MP_BUFFER_RW);
    if (bufinfo.typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("Expecting float array on X"));
    }

    return mp_obj_new_float(tdc_predict(self->root,bufinfo.buf));
}
STATIC MP_DEFINE_CONST_FUN_OBJ_2(emltreeslearner_EmlTreeLearner_predict_obj, emltreeslearner_EmlTreeLearner_predict);

STATIC mp_obj_t emltreeslearner_EmlTreeLearner_score(size_t n_args, const mp_obj_t *args) {
    mp_obj_t self_in = args[0];
    mp_obj_t X = args[1];
    mp_obj_t Y = args[2];
    mp_obj_t row_stride = args[3];

    emltreeslearner_EmlTreeLearner_obj_t *self = MP_OBJ_TO_PTR(self_in);

    if(!mp_obj_is_int(row_stride)){
        mp_raise_ValueError(MP_ERROR_TEXT("Expecting int array on row_stride"));
    }
    uint32_t row_str = mp_obj_get_int(row_stride);

    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(X, &bufinfo, MP_BUFFER_RW);
    if (bufinfo.typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("Expecting float array on Y"));
    }
    
    mp_buffer_info_t bufinfo0;
    mp_get_buffer_raise(Y, &bufinfo0, MP_BUFFER_RW);
    if (bufinfo0.typecode != 'f') {
        mp_raise_ValueError(MP_ERROR_TEXT("Expecting float array on Y"));
    }

    if((mp_uint_t)((bufinfo.len)/sizeof(float)/row_str)!=(mp_uint_t)((bufinfo0.len)/sizeof(float))){
        mp_raise_ValueError(MP_ERROR_TEXT("X and Y have different number of rows"));    
    }

    mp_obj_t rslt = mp_obj_new_float(tdc_score(self->root,(float*)bufinfo.buf,(float*)bufinfo0.buf,(bufinfo0.len)/sizeof(float),row_str));
    return rslt;
}
STATIC MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(emltreeslearner_EmlTreeLearner_score_obj, 4, 4, emltreeslearner_EmlTreeLearner_score);

// STATIC mp_obj_t emltreeslearner_EmlTreeLearner_serialize(mp_obj_t self_in) {
//     emltreeslearner_EmlTreeLearner_obj_t *self = MP_OBJ_TO_PTR(self_in);
//     uint32_t out_data_size = 100;
//     char* out_data = (char*)tdc_malloc(out_data_size);
//     char** outdatatr=&out_data;
//     struct Node** roots=(struct Node**)tdc_malloc(sizeof(struct Node*));
//     roots[0]=self->root;
    
//     tdc_save(outdatatr,&out_data_size,roots,1);
//     mp_obj_t out_str = mp_obj_new_str(out_data, strlen(out_data));

//     tdc_free(out_data,out_data_size);
//     tdc_free(roots,sizeof(struct Node*));

//     return out_str;
// }
// STATIC MP_DEFINE_CONST_FUN_OBJ_1(emltreeslearner_EmlTreeLearner_serialize_obj, emltreeslearner_EmlTreeLearner_serialize);

// STATIC mp_obj_t emltreeslearner_EmlTreeLearner_load(mp_obj_t self_in,mp_obj_t obj_string) {
//     emltreeslearner_EmlTreeLearner_obj_t *self = MP_OBJ_TO_PTR(self_in);

//     if(!mp_obj_is_str(obj_string)){
//         mp_raise_ValueError(MP_ERROR_TEXT("Expecting a string on input"));
//     }

//     size_t str_len;
//     char *str_data = (char*)mp_obj_str_get_data(obj_string, &str_len);

//     uint32_t rootsSize=0;
//     struct Node** roots = tdc_load(str_data,&rootsSize);
//     if(rootsSize!=1){
//         tdc_free(roots,sizeof(struct Node**));
//         mp_raise_ValueError(MP_ERROR_TEXT("Expecting a tree with one root"));
//     }
//     tdc_cleanup(&self->root);
//     self->root = roots[0];
//     return mp_const_none;
// }
// STATIC MP_DEFINE_CONST_FUN_OBJ_2(emltreeslearner_EmlTreeLearner_load_obj, emltreeslearner_EmlTreeLearner_load);

mp_map_elem_t emltreeslearner_EmlTreeLearner_locals_dict_table[4];
STATIC MP_DEFINE_CONST_DICT(emltreeslearner_EmlTreeLearner_locals_dict, emltreeslearner_EmlTreeLearner_locals_dict_table);

// This is the entry point and is called when the module is imported
mp_obj_t mpy_init(mp_obj_fun_bc_t *self, size_t n_args, size_t n_kw, mp_obj_t *args) {
    // This must be first, it sets up the globals dict and other things
    MP_DYNRUNTIME_INIT_ENTRY

    mp_store_global(MP_QSTR___name__, MP_OBJ_FROM_PTR(&MP_QSTR_emltreeslearner));
    mp_store_global(MP_QSTR_EmlTreeLearner, MP_OBJ_FROM_PTR(&emltreeslearner_type_EmlTreeLearner));

    emltreeslearner_type_EmlTreeLearner.base.type = (void*)&mp_fun_table.type_type;
    emltreeslearner_type_EmlTreeLearner.flags = MP_TYPE_FLAG_NONE;
    emltreeslearner_type_EmlTreeLearner.name = MP_QSTR_EmlTreeLearner;

    MP_OBJ_TYPE_SET_SLOT(&emltreeslearner_type_EmlTreeLearner, make_new, &emltreeslearner_EmlTreeLearner_make_new, 0);

    emltreeslearner_EmlTreeLearner_locals_dict_table[0] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_fit), MP_OBJ_FROM_PTR(&emltreeslearner_EmlTreeLearner_fit_obj) };
    emltreeslearner_EmlTreeLearner_locals_dict_table[1] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_plot), MP_OBJ_FROM_PTR(&emltreeslearner_EmlTreeLearner_plot_obj) };
    emltreeslearner_EmlTreeLearner_locals_dict_table[2] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_predict), MP_OBJ_FROM_PTR(&emltreeslearner_EmlTreeLearner_predict_obj) };
    emltreeslearner_EmlTreeLearner_locals_dict_table[3] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_score), MP_OBJ_FROM_PTR(&emltreeslearner_EmlTreeLearner_score_obj) };

    MP_OBJ_TYPE_SET_SLOT(&emltreeslearner_type_EmlTreeLearner, locals_dict, (void*)&emltreeslearner_EmlTreeLearner_locals_dict, 6);

    // This must be last, it restores the globals dict
    MP_DYNRUNTIME_INIT_EXIT
}

