function log_fatal(verbosity, msg...)
    verbosity >= -1 && warn(msg...)
end

function log_warning(verbosity, msg...)
    verbosity >= 0 && warn(msg...)
end

function log_info(verbosity, msg...)
    verbosity >= 1 && print(msg...)
end

function log_debug(verbosity, msg...)
    verbosity >= 2 && print(msg...)
end

function shrinkresults!(results, last_retained_iter::Integer)
    for test_key in keys(results)
        test = results[test_key]
        for metric_key in keys(test)
            test[metric_key] = test[metric_key][1:last_retained_iter]
        end
    end
    return nothing
end
