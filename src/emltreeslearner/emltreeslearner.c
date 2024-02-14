// Include MicroPython API.
#include "py/dynruntime.h"

#define DTR_DEBUG_PRINT(X) mp_printf(&mp_plat_print,"%s",X);
#define DTR_DEBUG_PRINTLN(X) mp_printf(&mp_plat_print,"%s\n\r",X);
#define DTR_DEBUG_PRINT_F(X) mp_printf(&mp_plat_print,"%f",(double)X);
#define DTR_DEBUG_PRINT_U(X) mp_printf(&mp_plat_print,"%lu",(uint32_t)X);
#define tdc_malloc(num_of_bytes) m_malloc(num_of_bytes)
#define tdc_realloc(ptr,old_num_bytes,new_num_bytes) m_realloc(ptr,new_num_bytes)
#define tdc_free(ptr,num_of_bytes) m_free(ptr)

#include "/home/alex/Documents/PlatformIO/Projects/decisionTreeC/lib/TinyDecisionTreeClassifier/src/TinyDecisionTreeClassifierC/TinyDecisionTreeClassifierC.h"
#include "/home/alex/Documents/PlatformIO/Projects/decisionTreeC/lib/TinyDecisionTreeClassifier/src/TinyDecisonTreeClassifierLoaderC/TinyDecisionTreeClassifierLoaderC.h"

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
    // All objects start with the base.
    mp_obj_base_t base;
    // Everything below can be thought of as instance attributes, but they
    // cannot be accessed by MicroPython code directly. In this example we
    // store the time at which the object was created.
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

STATIC mp_obj_t emltreeslearner_EmlTreeLearner_fit(mp_obj_t self_in, mp_obj_t X, mp_obj_t Y) {
    emltreeslearner_EmlTreeLearner_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if(!mp_obj_is_type(X,&mp_type_list)){
        mp_raise_ValueError(MP_ERROR_TEXT("Expecting list on X"));
    }
    //TODO arguments check
    if (mp_obj_is_type(Y, &mp_type_array)) {
        mp_obj_t *Xsamples;
        size_t XsamplesSize;

        mp_buffer_info_t bufinfo;
        mp_buffer_info_t bufinfo0;
        mp_get_buffer_raise(Y, &bufinfo0, MP_BUFFER_RW);
        if (bufinfo0.typecode != 'f') {
            mp_raise_ValueError(MP_ERROR_TEXT("Expecting float array on Y"));
        }
        mp_obj_list_get(X, &XsamplesSize, &Xsamples);

        if((mp_uint_t)((bufinfo0.len)/sizeof(float))!=XsamplesSize){
            mp_raise_ValueError(MP_ERROR_TEXT("X and Y have different number of rows"));    
        }
        float**XIn = tdc_malloc(XsamplesSize * sizeof(float *));
        if (XIn == NULL) {
            mp_raise_msg(&mp_type_MemoryError, MP_ERROR_TEXT("Failed to allocate memory for X pointer array"));
            return NULL;
        }
        for (size_t i = 0; i < XsamplesSize; i++) {
            // Get the inner buffer
            mp_get_buffer_raise(Xsamples[i], &bufinfo, MP_BUFFER_RW);
            if (bufinfo.typecode != 'f') {
                mp_raise_ValueError(MP_ERROR_TEXT("Expecting float array"));
            }
            XIn[i]=bufinfo.buf;
            }

        mp_get_buffer_raise(Xsamples[0], &bufinfo, MP_BUFFER_RW);
        mp_uint_t cols = (mp_uint_t)((bufinfo.len)/sizeof(float));

        float * Ybuf = bufinfo0.buf;
        tdc_fit_h(&self->root,XIn,Ybuf,XsamplesSize,cols,self->max_depth,self->min_sample_split);

        tdc_free(XIn,XsamplesSize * sizeof(float *));
    } else {
        if(!mp_obj_is_type(Y,&mp_type_list)){
            mp_raise_ValueError(MP_ERROR_TEXT("Expecting list or array of floats on Y"));
        }
        mp_obj_t *Xsamples;
        size_t XsamplesSize;

        mp_obj_t *Ysamples;
        size_t YsamplesSize;

        mp_buffer_info_t bufinfo;

        mp_obj_list_get(X, &XsamplesSize, &Xsamples);
        mp_obj_list_get(Y, &YsamplesSize, &Ysamples);

        if(YsamplesSize!=XsamplesSize){
            mp_raise_ValueError(MP_ERROR_TEXT("X and Y have different number of rows"));    
        }

        float**XIn = tdc_malloc(XsamplesSize * sizeof(float *));
        if (XIn == NULL) {
            mp_raise_msg(&mp_type_MemoryError, MP_ERROR_TEXT("Failed to allocate memory for X pointer array"));
            return NULL;
        }

        float**YIn = tdc_malloc(YsamplesSize * sizeof(float *));
        if (YIn == NULL) {
            mp_raise_msg(&mp_type_MemoryError, MP_ERROR_TEXT("Failed to allocate memory for Y pointer array"));
            return NULL;
        }

        for (size_t i = 0; i < XsamplesSize; i++) {
            // Get the inner buffer
            mp_get_buffer_raise(Xsamples[i], &bufinfo, MP_BUFFER_RW);
            if (bufinfo.typecode != 'f') {
                mp_raise_ValueError(MP_ERROR_TEXT("Expecting float array"));
            }
            XIn[i]=bufinfo.buf;

            mp_get_buffer_raise(Ysamples[i], &bufinfo, MP_BUFFER_RW);
            if (bufinfo.typecode != 'f') {
                mp_raise_ValueError(MP_ERROR_TEXT("Expecting float array"));
            }
            YIn[i]=bufinfo.buf;
            }

        mp_get_buffer_raise(Xsamples[0], &bufinfo, MP_BUFFER_RW);
        mp_uint_t cols = (mp_uint_t)((bufinfo.len)/sizeof(float));

        tdc_fit(&self->root,XIn,YIn,XsamplesSize,cols,self->max_depth,self->min_sample_split);

        tdc_free(XIn,XsamplesSize * sizeof(float *));
        tdc_free(YIn,YsamplesSize * sizeof(float *));
    }
    return mp_const_none;
}
STATIC MP_DEFINE_CONST_FUN_OBJ_3(emltreeslearner_EmlTreeLearner_fit_obj, emltreeslearner_EmlTreeLearner_fit);

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

STATIC mp_obj_t emltreeslearner_EmlTreeLearner_score(mp_obj_t self_in, mp_obj_t X, mp_obj_t Y) {
    emltreeslearner_EmlTreeLearner_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if(!mp_obj_is_type(X,&mp_type_list)){
        mp_raise_ValueError(MP_ERROR_TEXT("Expecting list on X"));
    }

    if(mp_obj_is_type(Y,&mp_type_array)){
        mp_obj_t *Xsamples;
        size_t XsamplesSize;

        mp_buffer_info_t bufinfo;
        mp_buffer_info_t bufinfo0;
        mp_get_buffer_raise(Y, &bufinfo0, MP_BUFFER_RW);
        if (bufinfo0.typecode != 'f') {
            mp_raise_ValueError(MP_ERROR_TEXT("Expecting float array on Y"));
        }
        mp_obj_list_get(X, &XsamplesSize, &Xsamples);

        if((mp_uint_t)((bufinfo0.len)/sizeof(float))!=XsamplesSize){
            mp_raise_ValueError(MP_ERROR_TEXT("X and Y have different number of rows"));    
        }
        float**XIn = tdc_malloc(XsamplesSize * sizeof(float *));
        if (XIn == NULL) {
            mp_raise_msg(&mp_type_MemoryError, MP_ERROR_TEXT("Failed to allocate memory for X pointer array"));
            return NULL;
        }
        for (size_t i = 0; i < XsamplesSize; i++) {
            // Get the inner buffer
            mp_get_buffer_raise(Xsamples[i], &bufinfo, MP_BUFFER_RW);
            if (bufinfo.typecode != 'f') {
                mp_raise_ValueError(MP_ERROR_TEXT("Expecting float array"));
            }
            XIn[i]=bufinfo.buf;
            }

        mp_get_buffer_raise(Xsamples[0], &bufinfo, MP_BUFFER_RW);

        float * Ybuf = bufinfo0.buf;
        mp_obj_t rslt = mp_obj_new_float(tdc_score_h(self->root,XIn,Ybuf,XsamplesSize));
        tdc_free(XIn,XsamplesSize * sizeof(float *));
        return rslt;
    }
    else{
        if(!mp_obj_is_type(Y,&mp_type_list)){
            mp_raise_ValueError(MP_ERROR_TEXT("Expecting list or array of floats on Y"));
        }
        mp_obj_t *Xsamples;
        size_t XsamplesSize;
        mp_obj_t *Ysamples;
        size_t YsamplesSize;
        mp_buffer_info_t bufinfo;
        mp_obj_list_get(X, &XsamplesSize, &Xsamples);
        mp_obj_list_get(Y, &YsamplesSize, &Ysamples);

        if(YsamplesSize!=XsamplesSize){
            mp_raise_ValueError(MP_ERROR_TEXT("X and Y have different number of rows"));    
        }
        float**XIn = tdc_malloc(XsamplesSize * sizeof(float *));
        if (XIn == NULL) {
            mp_raise_msg(&mp_type_MemoryError, MP_ERROR_TEXT("Failed to allocate memory for X pointer array"));
            return NULL;
        }
        float**YIn = tdc_malloc(YsamplesSize * sizeof(float *));
        if (YIn == NULL) {
            mp_raise_msg(&mp_type_MemoryError, MP_ERROR_TEXT("Failed to allocate memory for Y pointer array"));
            return NULL;
        }
        for (size_t i = 0; i < XsamplesSize; i++) {
            // Get the inner buffer
            mp_get_buffer_raise(Xsamples[i], &bufinfo, MP_BUFFER_RW);
            if (bufinfo.typecode != 'f') {
                mp_raise_ValueError(MP_ERROR_TEXT("Expecting float array"));
            }        
            XIn[i]=bufinfo.buf;
            mp_get_buffer_raise(Ysamples[i], &bufinfo, MP_BUFFER_RW);
            if (bufinfo.typecode != 'f') {
                mp_raise_ValueError(MP_ERROR_TEXT("Expecting float array"));
            }
            YIn[i]=bufinfo.buf;
            }    
        mp_get_buffer_raise(Xsamples[0], &bufinfo, MP_BUFFER_RW);

        tdc_free(XIn,XsamplesSize * sizeof(float *));
        tdc_free(YIn,YsamplesSize * sizeof(float *));

        return mp_obj_new_float(tdc_score(self->root,XIn,YIn,XsamplesSize));
    }
}
STATIC MP_DEFINE_CONST_FUN_OBJ_3(emltreeslearner_EmlTreeLearner_score_obj, emltreeslearner_EmlTreeLearner_score);

STATIC mp_obj_t emltreeslearner_EmlTreeLearner_serialize(mp_obj_t self_in) {
    emltreeslearner_EmlTreeLearner_obj_t *self = MP_OBJ_TO_PTR(self_in);
    uint32_t out_data_size = 100;
    char* out_data = (char*)tdc_malloc(out_data_size);
    char** outdatatr=&out_data;
    struct Node** roots=(struct Node**)tdc_malloc(sizeof(struct Node*));
    roots[0]=self->root;
    
    tdc_save(outdatatr,&out_data_size,roots,1);
    mp_obj_t out_str = mp_obj_new_str(out_data, strlen(out_data));

    tdc_free(out_data,out_data_size);
    tdc_free(roots,sizeof(struct Node*));

    return out_str;
}
STATIC MP_DEFINE_CONST_FUN_OBJ_1(emltreeslearner_EmlTreeLearner_serialize_obj, emltreeslearner_EmlTreeLearner_serialize);

STATIC mp_obj_t emltreeslearner_EmlTreeLearner_load(mp_obj_t self_in,mp_obj_t obj_string) {
    emltreeslearner_EmlTreeLearner_obj_t *self = MP_OBJ_TO_PTR(self_in);

    if(!mp_obj_is_str(obj_string)){
        mp_raise_ValueError(MP_ERROR_TEXT("Expecting a string on input"));
    }

    size_t str_len;
    char *str_data = (char*)mp_obj_str_get_data(obj_string, &str_len);

    uint32_t rootsSize=0;
    struct Node** roots = tdc_load(str_data,&rootsSize);
    if(rootsSize!=1){
        tdc_free(roots,sizeof(struct Node**));
        mp_raise_ValueError(MP_ERROR_TEXT("Expecting a tree with one root"));
    }
    tdc_cleanup(&self->root);
    self->root = roots[0];
    return mp_const_none;
}
STATIC MP_DEFINE_CONST_FUN_OBJ_2(emltreeslearner_EmlTreeLearner_load_obj, emltreeslearner_EmlTreeLearner_load);

STATIC mp_obj_t emltreeslearner_load_model(mp_obj_t self_in,mp_obj_t obj_string) {
    return emltreeslearner_EmlTreeLearner_load(self_in,obj_string);
}
STATIC MP_DEFINE_CONST_FUN_OBJ_2(emltreeslearner_load_model_obj, emltreeslearner_load_model);

// This collects all methods and other static class attributes of the EmlTreeLearner.
// The table structure is similar to the module table, as detailed below.
// STATIC mp_rom_map_elem_t emltreeslearner_EmlTreeLearner_locals_dict_table[] = {
//     {MP_ROM_QSTR(MP_QSTR_fit),MP_ROM_PTR(&emltreeslearner_EmlTreeLearner_fit_obj)},
//     {MP_ROM_QSTR(MP_QSTR_plot),MP_ROM_PTR(&emltreeslearner_EmlTreeLearner_plot_obj)},
//     {MP_ROM_QSTR(MP_QSTR_predict),MP_ROM_PTR(&emltreeslearner_EmlTreeLearner_predict_obj)},
//     {MP_ROM_QSTR(MP_QSTR_score),MP_ROM_PTR(&emltreeslearner_EmlTreeLearner_score_obj)},
//     {MP_ROM_QSTR(MP_QSTR_serialize),MP_ROM_PTR(&emltreeslearner_EmlTreeLearner_serialize_obj)},
//     {MP_ROM_QSTR(MP_QSTR_load),MP_ROM_PTR(&emltreeslearner_EmlTreeLearner_load_obj)},
// };
// STATIC MP_DEFINE_CONST_DICT(emltreeslearner_EmlTreeLearner_locals_dict, emltreeslearner_EmlTreeLearner_locals_dict_table);

// const mp_obj_module_t emltreeslearner = {
//     .base = { &mp_type_module },
//     .globals = (mp_obj_dict_t *)&emltreeslearner_module_globals,
// };

mp_map_elem_t emltreeslearner_EmlTreeLearner_locals_dict_table[6];
STATIC MP_DEFINE_CONST_DICT(emltreeslearner_EmlTreeLearner_locals_dict, emltreeslearner_EmlTreeLearner_locals_dict_table);

// This is the entry point and is called when the module is imported
mp_obj_t mpy_init(mp_obj_fun_bc_t *self, size_t n_args, size_t n_kw, mp_obj_t *args) {
    // This must be first, it sets up the globals dict and other things
    MP_DYNRUNTIME_INIT_ENTRY

    mp_store_global(MP_QSTR___name__, MP_OBJ_FROM_PTR(&MP_QSTR_emltreeslearner));
    mp_store_global(MP_QSTR_EmlTreeLearner, MP_OBJ_FROM_PTR(&emltreeslearner_type_EmlTreeLearner));
    mp_store_global(MP_QSTR_load_model, MP_OBJ_FROM_PTR(&emltreeslearner_load_model_obj));

    emltreeslearner_type_EmlTreeLearner.base.type = (void*)&mp_fun_table.type_type;
    emltreeslearner_type_EmlTreeLearner.flags = MP_TYPE_FLAG_NONE;
    emltreeslearner_type_EmlTreeLearner.name = MP_QSTR_EmlTreeLearner;

    MP_OBJ_TYPE_SET_SLOT(&emltreeslearner_type_EmlTreeLearner, make_new, &emltreeslearner_EmlTreeLearner_make_new, 0);

    emltreeslearner_EmlTreeLearner_locals_dict_table[0] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_fit), MP_OBJ_FROM_PTR(&emltreeslearner_EmlTreeLearner_fit_obj) };
    emltreeslearner_EmlTreeLearner_locals_dict_table[1] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_plot), MP_OBJ_FROM_PTR(&emltreeslearner_EmlTreeLearner_plot_obj) };
    emltreeslearner_EmlTreeLearner_locals_dict_table[2] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_predict), MP_OBJ_FROM_PTR(&emltreeslearner_EmlTreeLearner_predict_obj) };
    emltreeslearner_EmlTreeLearner_locals_dict_table[3] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_score), MP_OBJ_FROM_PTR(&emltreeslearner_EmlTreeLearner_score_obj) };
    emltreeslearner_EmlTreeLearner_locals_dict_table[4] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_serialize), MP_OBJ_FROM_PTR(&emltreeslearner_EmlTreeLearner_serialize_obj) };
    emltreeslearner_EmlTreeLearner_locals_dict_table[5] = (mp_map_elem_t){ MP_OBJ_NEW_QSTR(MP_QSTR_load), MP_OBJ_FROM_PTR(&emltreeslearner_EmlTreeLearner_load_obj) };

    MP_OBJ_TYPE_SET_SLOT(&emltreeslearner_type_EmlTreeLearner, locals_dict, (void*)&emltreeslearner_EmlTreeLearner_locals_dict, 6);

    // trees_builder_type.base.type = (void*)&mp_fun_table.type_type;
    // trees_builder_type.flags = MP_TYPE_FLAG_ITER_IS_CUSTOM;
    // trees_builder_type.name = MP_QSTR_emltrees;
    // methods
//     {MP_ROM_QSTR(MP_QSTR_fit),MP_ROM_PTR(&emltreeslearner_EmlTreeLearner_fit_obj)},
//     {MP_ROM_QSTR(MP_QSTR_plot),MP_ROM_PTR(&emltreeslearner_EmlTreeLearner_plot_obj)},
//     {MP_ROM_QSTR(MP_QSTR_predict),MP_ROM_PTR(&emltreeslearner_EmlTreeLearner_predict_obj)},
//     {MP_ROM_QSTR(MP_QSTR_score),MP_ROM_PTR(&emltreeslearner_EmlTreeLearner_score_obj)},
//     {MP_ROM_QSTR(MP_QSTR_serialize),MP_ROM_PTR(&emltreeslearner_EmlTreeLearner_serialize_obj)},
//     {MP_ROM_QSTR(MP_QSTR_load),MP_ROM_PTR(&emltreeslearner_EmlTreeLearner_load_obj)},


    // This must be last, it restores the globals dict
    MP_DYNRUNTIME_INIT_EXIT
}

