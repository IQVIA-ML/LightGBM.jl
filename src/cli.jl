const tempdir = "lightgbm_temp"

# Fit the `estimator` using the CLI of LightGBM.
function cli_fit{TX<:Real,Ty<:Real}(estimator::LGBMEstimator, X::Array{TX,2}, y::Array{Ty,1},
                                    test::Tuple{Array{TX,2},Array{Ty,1}}...;
                                    verbosity::Integer = 1)
    rm_tempdir = false # Don't remove an existing directory, just in case
    if !isdir(tempdir)
        mkdir(tempdir)
        rm_tempdir = true
    end

    conf = open("$(tempdir)/lightgbm.conf", "w")
    cli_prep_data(conf, X, y)
    cli_prep_test(conf, test)
    cli_prep_fit(estimator, conf)
    close(conf)

    results = Dict{String,Dict{String,Array{Float64,1}}}()
    open(`$(ENV["LIGHTGBM"]) config=$(pwd())/$(tempdir)/lightgbm.conf`, "r") do pipe
        stopped = false
        while isopen(pipe)
            output = readline(pipe)
            printoutput(output, verbosity)
            stopped = cli_processoutput!(results, output, estimator, stopped)
        end
    end
    estimator.model = readlines("$(tempdir)/model.txt")

    if rm_tempdir
        rm(tempdir, force = true, recursive = true)
    else
        info("Did not remove the temporary working directory $(pwd())/$(tempdir)/ because it already existed.")
    end

    return results
end

# Predict using the CLI of LightGBM.
function cli_predict{TX<:Real}(estimator::LGBMEstimator, X::Array{TX,2}; verbosity::Integer = 1)
    @assert(length(estimator.model) > 0, "Estimator does not contain a fitted model.")

    rm_tempdir = false # Don't remove an existing directory, just in case
    if !isdir(tempdir)
        mkdir(tempdir)
        rm_tempdir = true
    end

    conf = open("$(tempdir)/lightgbm.conf", "w")
    cli_prep_data(conf, X)
    cli_prep_predict(estimator, conf)
    close(conf)

    open(`$(ENV["LIGHTGBM"]) config=$(pwd())/$(tempdir)/lightgbm.conf`, "r") do pipe
        while isopen(pipe)
            output = readline(pipe)
            printoutput(output, verbosity)
        end
    end
    results = vec(readcsv("$(tempdir)/results.txt"))

    if rm_tempdir
        rm(tempdir, force = true, recursive = true)
    else
        info("Did not remove the temporary working directory $(pwd())/$(tempdir)/ because it already existed.")
    end

    return results
end

# Prints or ignores the output based on the `verbosity` setting.
function printoutput(output::String, verbosity::Integer)
    if startswith(output, "[LightGBM] [Info]")
        level = 1
    elseif startswith(output, "[LightGBM] [Warning]")
        level = 0
    elseif startswith(output, "[LightGBM] [Fatal]")
        level = -1
    elseif startswith(output, "[LightGBM] [Debug]")
        level = 2
    else
        level = -1
    end
    level <= verbosity && print(output)

    return nothing
end

# Parse the LightGBM `output` for test metrics and early stopping. Store test metrics in `results`.
# Shrink the `results` to the best iteration round when early stopping is detected.
function cli_processoutput!(results, output::String, estimator::LGBMEstimator, stopped::Bool)
    if startswith(output, "[LightGBM] [Info] Iteration: ")
        iter, test, metric, score = cli_parse_metrics(output)
        storeresults!(results, estimator, iter, test, metric, score)
    elseif startswith(output, "[LightGBM] [Info] Early stopping")
        best_iter = cli_parse_earlystop(output)
        shrinkresults!(results, best_iter)
    elseif startswith(output, "[LightGBM] [Info] Stopped training")
        stopped = true
    elseif stopped && contains(output, " seconds elapsed, finished iteration ")
        last_iter = cli_parse_stop(output)
        shrinkresults!(results, last_iter)
    end

    return stopped
end

# Parse the LightGBM test metrics `output` and return the iteration round, test set, metric name,
# and metric value.
function cli_parse_metrics(output)
    iter_start_idx = 30 # [LightGBM] [Info] Iteration: _
    iter_end_idx = searchindex(output, ',', iter_start_idx) - 1
    iter = parse(Int, output[iter_start_idx:iter_end_idx])

    test_start_idx = iter_end_idx + 3
    test_start_idx = last(search(output, "/$(tempdir)/", test_start_idx)) + 1
    test_end_idx = first(search(output, ".csv's ", test_start_idx)) - 1
    test = output[test_start_idx:test_end_idx]

    metric_start_idx = test_end_idx + 8
    metric_end_idx = search(output, ':', metric_start_idx) - 1
    metric = output[metric_start_idx:metric_end_idx]

    score_start_idx = metric_end_idx + 3
    score = float(output[score_start_idx:end])

    return iter, test, metric, score
end

# Parse the LightGBM early stopping `output` and return the best iteration round.
function cli_parse_earlystop(output)
    best_start_idx = last(search(output, "the best iteration round is ")) + 1
    best_iter = parse(Int, output[best_start_idx:end])

    return best_iter
end

# Parse the LightGBM stopping `output` and return the last completed iteration round.
function cli_parse_stop(output)
    last_start_idx = last(search(output, " seconds elapsed, finished iteration ")) + 1
    last_iter = parse(Int, output[last_start_idx:end])

    return last_iter - 1
end

# Write the `data` to `filename` as an Array{Float32,2}.
function write_file{T<:Real}(filename::String, data::Array{T,2})
    file = open(filename, "w")
    writecsv(file, convert(Array{Float32,2}, data))
    close(file)

    return nothing
end

# Write `X` and `y` to `filename` as an Array{Float32,2} with `y` as the first column.
function write_file{TX<:Real,Ty<:Real}(filename::String, X::Array{TX,2}, y::Array{Ty,1})
    data = hcat(convert(Array{Float32,1}, y), convert(Array{Float32,2}, X))
    write_file(filename, data)

    return nothing
end

# Write `data` to the temporary working directory and add the data entry to `conf`.
function cli_prep_data(conf, data...)
    filename = "$(pwd())/$(tempdir)/data.csv"
    write(conf, "data = ", filename, "\n")
    write_file(filename, data...)

    return nothing
end

# Write the data entries of `test` to the temporary working directory and add them as valid entries
# valid entries to `conf`.
function cli_prep_test(conf, test)
    n_tests = length(test)

    if n_tests == 1
        filename = "$(pwd())/$(tempdir)/test_1.csv"
        write(conf, "valid = ", filename, "\n")
        write_file(filename, test[1][1], test[1][2])
    elseif n_tests > 1
        filename = "$(pwd())/$(tempdir)/test_1.csv"
        write(conf, "valid = ", filename)
        write_file(filename, test[1][1], test[1][2])
        for test_idx in 2:n_tests
            filename = "$(pwd())/$(tempdir)/test_$(test_idx).csv"
            write(conf, ",", filename)
            write_file(filename, test[test_idx][1], test[test_idx][2])
        end
        write(conf, "\n")
    end

    return nothing
end

# Add prediction entries to `conf` and write `estimator.model` to disk.
function cli_prep_predict(estimator::LGBMEstimator, conf)
    write(conf, "task = prediction\n")
    write(conf, "input_model = $(pwd())/$(tempdir)/model.txt\n")
    write(conf, "output_result = $(pwd())/$(tempdir)/results.txt\n")

    for field in fieldnames(estimator)
        if in(field, (:is_sigmoid,))
            write(conf, string(field), " = ", string(getfield(estimator, field)), "\n")
        end
    end

    model = open("$(tempdir)/model.txt", "w")
    write(model, estimator.model)
    close(model)

    return nothing
end

# Add training entries to `conf`.
function cli_prep_fit(estimator::LGBMEstimator, conf)
    write(conf, "task = train\n")
    write(conf, "output_model = $(pwd())/$(tempdir)/model.txt\n")

    for field in fieldnames(estimator)
        if in(field, (:machine_list_file, :init_score)) # Escape tempdir for user-specified filenames
            if getfield(estimator, field) != ""
                write(conf, string(field), " = ", pwd(), "/", string(getfield(estimator, field)),
                      "\n")
            end
        elseif in(field, (:valid, :label_gain, :metric, :ndcg_at)) # Comma-seperate array entries
            n_entries = length(getfield(estimator, field))
            if n_entries == 1
                write(conf, string(field), " = ", string(getfield(estimator, field)[1]), "\n")
            elseif n_entries > 1
                write(conf, string(field), " = ", string(getfield(estimator, field)[1]))
                for entry_idx in 2:n_entries
                    write(conf, ",", string(getfield(estimator, field)[entry_idx]))
                end
                write(conf, "\n")
            end
        elseif !in(field, (:model,)) # These fields aren't used tor training.
            write(conf, string(field), " = ", string(getfield(estimator, field)), "\n")
        end
    end

    return nothing
end

function storeresults!(results, estimator::LGBMEstimator, iter::Integer, test::String,
                       metric::String, score::Float64)
    if !haskey(results, test)
        results[test] = Dict{String,Array{Float64,1}}()
        results[test][metric] = Array(Float64, estimator.num_iterations)
    elseif !haskey(results[test], metric)
        results[test][metric] = Array(Float64, estimator.num_iterations)
    end
    results[test][metric][iter] = score

    return nothing
end
