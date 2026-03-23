#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct callback_data {
    std::vector<uint8_t> data;
    int decode_calls = 0;
};

static bool has_name(const ggml_tensor * t, const char * want) {
    return t && std::strncmp(t->name, want, std::strlen(want)) == 0;
}

static void print_i32_tensor(const ggml_tensor * t, uint8_t * data, int layer_index, int decode_call_index) {
    if (t->type != GGML_TYPE_I32) {
        return;
    }

    const int64_t n_expert_used = t->ne[0];
    const int64_t n_tokens = t->ne[1];

    std::printf("expert_trace layer=%d decode_call=%d shape=[%lld,%lld]\n",
            layer_index, decode_call_index,
            (long long) n_expert_used, (long long) n_tokens);

    for (int64_t tok = 0; tok < n_tokens; ++tok) {
        std::printf("  token=%lld experts=", (long long) tok);
        for (int64_t ex = 0; ex < n_expert_used; ++ex) {
            const size_t offset = tok * t->nb[1] + ex * t->nb[0];
            const int32_t expert_id = *(int32_t *) (data + offset);
            std::printf("%s%d", ex == 0 ? "[" : ",", expert_id);
        }
        std::printf("]\n");
    }
}

static void print_f32_like_tensor(const ggml_tensor * t, uint8_t * data, int layer_index, int decode_call_index) {
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) {
        return;
    }

    const int64_t n_expert_used = t->ne[1];
    const int64_t n_tokens = t->ne[2];

    std::printf("expert_weight_trace layer=%d decode_call=%d shape=[%lld,%lld]\n",
            layer_index, decode_call_index,
            (long long) n_expert_used, (long long) n_tokens);

    for (int64_t tok = 0; tok < n_tokens; ++tok) {
        std::printf("  token=%lld weights=", (long long) tok);
        for (int64_t ex = 0; ex < n_expert_used; ++ex) {
            const size_t offset = tok * t->nb[2] + ex * t->nb[1];
            float weight = 0.0f;
            if (t->type == GGML_TYPE_F16) {
                weight = ggml_fp16_to_fp32(*(ggml_fp16_t *) (data + offset));
            } else {
                weight = *(float *) (data + offset);
            }
            std::printf("%s%.6f", ex == 0 ? "[" : ",", weight);
        }
        std::printf("]\n");
    }
}

static bool ggml_trace_experts(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (callback_data *) user_data;

    const bool is_topk = has_name(t, "ffn_moe_topk");
    const bool is_weights = has_name(t, "ffn_moe_weights_norm") || has_name(t, "ffn_moe_weights");

    if (!is_topk && !is_weights) {
        return false;
    }

    if (ask) {
        return true;
    }

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    uint8_t * tensor_data = nullptr;

    if (is_host) {
        tensor_data = (uint8_t *) t->data;
    } else {
        const auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
        tensor_data = cb_data->data.data();
    }

    int layer_index = -1;
    if (const char * last_dot = std::strrchr(t->name, '-')) {
        layer_index = std::atoi(last_dot + 1);
    }

    if (is_topk) {
        print_i32_tensor(t, tensor_data, layer_index, cb_data->decode_calls);
    } else {
        print_f32_like_tensor(t, tensor_data, layer_index, cb_data->decode_calls);
    }
    return true;
}

static bool run(llama_context * ctx, const gpt_params & params) {
    const bool add_bos = llama_should_add_bos_token(llama_get_model(ctx));
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size(), 0, 0))) {
        std::fprintf(stderr, "%s: failed to eval\n", __func__);
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    callback_data cb_data;
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }

    print_build_info();

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval = ggml_trace_experts;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    llama_init_result init = llama_init_from_gpt_params(params);
    llama_model * model = init.model;
    llama_context * ctx = init.context;

    if (model == nullptr || ctx == nullptr) {
        std::fprintf(stderr, "%s: failed to init\n", __func__);
        return 1;
    }

    std::fprintf(stderr, "\n%s\n", gpt_params_get_system_info(params).c_str());

    const bool ok = run(ctx, params);
    llama_print_timings(ctx);

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return ok ? 0 : 1;
}
