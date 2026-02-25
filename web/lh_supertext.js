import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Helper function to create a resizable DOM widget
function setupSuperTextWidget(node, widgetName, inputName, app) {
    const widget = node.widgets?.find((w) => w.name === widgetName);
    // [Fix] Don't return early if inputEl is missing yet. 
    // Instead, we will retry applying styles in the draw loop.
    if (!widget) return; 

    const w = widget;
    
    // Function to apply styles (idempotent)
    const applyStyles = () => {
        if (!w.inputEl || w._lh_styles_applied) return;
        
        // Custom styling
        w.inputEl.style.border = "1px solid #444"; 
        w.inputEl.style.borderRadius = "4px";
        w.inputEl.style.padding = "6px"; 
        w.inputEl.style.lineHeight = "1.4"; // Restore readable line height
        // Remove bold as it makes text harder to read
        w.inputEl.style.fontWeight = "normal"; 
        
        // [Revert] No forced colors. Let ComfyUI handle theme.
        w.inputEl.style.removeProperty("color");
        w.inputEl.style.removeProperty("background-color");
        
        // Enable Resizing
        w.inputEl.style.resize = "vertical";
        w.inputEl.style.overflowY = "auto";
        
        // Selection styles
        w.inputEl.style.userSelect = "text";
        w.inputEl.style.webkitUserSelect = "text";
        
        w._lh_styles_applied = true;
    };

    // Try applying immediately (might fail if DOM not ready)
    applyStyles();
    w.inputEl.style.pointerEvents = "auto";
    w.inputEl.style.cursor = "text";

    // Update state based on connection
    const updateState = () => {
        // [Safety Check] Ensure inputEl exists and styles are applied before updating state
        if (!w.inputEl) return;
        if (!w._lh_styles_applied) applyStyles();

        // 1. Check the specific named input (e.g., "showtext", "instruction", "user_material")
        const input = node.inputs?.find(i => i.name === inputName);
        let isConnected = input && input.link !== null;

        // Helper to check if upstream node is active (not Muted/Bypassed)
        const isUpstreamActive = (linkId) => {
            if (linkId === null || linkId === undefined) return false;
            const link = app.graph.links[linkId];
            if (!link) return false;
            const originNode = app.graph.getNodeById(link.origin_id);
            if (!originNode) return false;
            
            // Mode 2 is Mute (Never run)
            // Mode 4 is Bypass (Pass through)
            if (originNode.mode === 2 || originNode.mode === 4) return false;
            return true;
        };

        // Check named input status
        let isMainInputActive = false;
        if (input && input.link !== null) {
            if (isUpstreamActive(input.link)) {
                isMainInputActive = true;
            }
        }

        // Check additional "text" input status (Specific to LH_SuperText node)
        let isExtraTextActive = false;
        if (node.type === "LH_SuperText") {
            const forceInput = node.inputs?.find(i => i.name === "text");
            if (forceInput && forceInput.link !== null) {
                if (isUpstreamActive(forceInput.link)) {
                    isExtraTextActive = true;
                }
            }
        }
        
        // Final decision: Lock if ANY active input is present
        // If it's LH_SuperText, we look at both 'showtext' (if exists) and 'text'
        // If it's UniversalAIChat, we only look at the named input (e.g. instruction)
        const shouldLock = isMainInputActive || isExtraTextActive;

        if (shouldLock) {
            if (!w.inputEl.readOnly) {
                w.inputEl.readOnly = true;
            }
            // Locked state (Read-Only) -> Dim it
            // Only use opacity to indicate state, matching native ComfyUI style
            w.inputEl.style.setProperty("opacity", "0.6", "important"); 
            w.inputEl.style.removeProperty("color");
            w.inputEl.style.removeProperty("background-color");
            w.inputEl.style.removeProperty("font-weight");
        } else {
            // Unlocked state (Editable)
            if (w.inputEl.readOnly) {
                w.inputEl.readOnly = false;
            }
            // Always enforce full opacity in unlocked state
            w.inputEl.style.setProperty("opacity", "1.0", "important");
            w.inputEl.style.removeProperty("color");
            w.inputEl.style.removeProperty("background-color");
            w.inputEl.style.removeProperty("font-weight");
        }
    };

    // Hook into connection changes
    const onConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function() {
        if (onConnectionsChange) onConnectionsChange.apply(this, arguments);
        updateState();
    };

    // Hook into draw foreground to update state on upstream mode changes (Mute/Bypass)
    const onDrawForeground = node.onDrawForeground;
    node.onDrawForeground = function(ctx) {
        if (onDrawForeground) onDrawForeground.apply(this, arguments);
        if (this.flags && this.flags.collapsed) return;
        
        // Restore 200ms throttle as requested
        // [Fix] Use unique timer property per widget to avoid conflict when multiple widgets on same node use this helper
        const timerProp = `_last_update_state_time_${widgetName}`;
        const now = Date.now();
        if (!this[timerProp] || (now - this[timerProp] > 200)) {
            updateState();
            this[timerProp] = now;
        }
    };

    // Initial state
    updateState();
}

// [New] Cleanup Function to remove unwanted inputs
    function cleanupInputs(node) {
        if (node.type !== "LH_SuperText") return;
        
        // Check if 'showtext' has an input slot
        const inputName = "showtext";
        const slotIdx = node.findInputSlot(inputName);
        
        if (slotIdx !== -1) {
             const input = node.inputs[slotIdx];
             if (!input.link) {
                 // Remove it if it's not connected.
                 // This forces the cleanup of the unwanted input slot.
                 node.removeInput(slotIdx);
             } else {
                 // If connected, it means user has an old workflow.
                 // To prevent confusion, we can HIDE the connection point visually?
                 // No, that's complex. Let's just remove it if user wants "cannot be connected".
                 // But removing connected input breaks user workflow.
                 // Let's assume user will manually reconnect to 'text' if needed.
                 // For now, only remove unconnected ones to be safe.
                 // But if user insists on "cannot be connected", we might need to be more aggressive?
                 // Let's keep it safe: Only remove unconnected.
             }
        }
    }

app.registerExtension({
	name: "Comfy.LoraHelper.Widgets",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 1. LH_SuperText (SuperText) Logic
        // Finalized: 2026-02-02 - DO NOT MODIFY
		if (nodeData.name === "LH_SuperText") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                setTimeout(() => {
                    setupSuperTextWidget(this, "showtext", "showtext", app);
                }, 50);

                // Cleanup unwanted inputs for SuperText
                setTimeout(() => {
                    cleanupInputs(this);
                }, 100);
                
                return r;
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = async function (message) {
                onExecuted?.apply(this, arguments);

                // Fix: Check for 'showtext' in message first (ComfyUI standard UI update)
                // Backend returns: {"ui": {"showtext": [text_to_process]}}
                const newText = message?.showtext?.[0] || message?.text?.[0];

                if (!newText) {
                    return;
                }

                const displayValue = newText;

                // Strategy: 
                // 1. Try to use existing 'showtext' widget if it's visible (not converted to input).
                // 2. If not found, use/create 'display_text' widget.

                let targetWidget = this.widgets?.find(w => w.name === "showtext");
                
                // If showtext is missing (converted to input), we need display_text
                if (!targetWidget) {
                    if (!this.widgets) this.widgets = [];
                    targetWidget = this.widgets.find(w => w.name === "display_text");
                    
                    if (!targetWidget) {
                        targetWidget = this.addWidget(
                            "text", 
                            "display_text", 
                            displayValue, 
                            () => {}, 
                            { multiline: true }
                        );
                        
                        // Async styling for the new widget
                        requestAnimationFrame(() => {
                             if (targetWidget.inputEl) {
                                const el = targetWidget.inputEl;
                                el.readOnly = true;
                                el.style.userSelect = "text";
                                el.style.webkitUserSelect = "text";
                                el.style.pointerEvents = "auto";
                                el.style.cursor = "text";
                                el.style.opacity = "0.9";
                                el.style.backgroundColor = "#1a1a1a";
                                el.style.color = "#e0e0e0";
                                el.style.border = "1px solid #444";
                                el.style.padding = "6px";
                                el.style.fontFamily = "monospace";
                                el.style.whiteSpace = "pre-wrap";
                                el.title = "Copyable (Ctrl+C)";
                             }
                        });
                    }
                } else {
                    // If showtext exists, ensure we remove any stale display_text to clean up
                    const displayIdx = this.widgets?.findIndex(w => w.name === "display_text");
                    if (displayIdx !== undefined && displayIdx >= 0) {
                        this.widgets.splice(displayIdx, 1);
                    }
                }

                // Update the found/created widget
                if (targetWidget) {
                    targetWidget.value = displayValue;
                    if (targetWidget.inputEl) {
                        targetWidget.inputEl.value = displayValue;
                    }
                    // Force UI update if needed
                    if (targetWidget.callback) {
                        targetWidget.callback(displayValue, app.canvas, this, app.canvas.graph_mouse, {});
                    }
                }

                // Force Redraw
                requestAnimationFrame(() => {
                    const sz = this.computeSize();
                    if (sz[0] < this.size[0]) sz[0] = this.size[0];
                    if (sz[1] < this.size[1]) sz[1] = this.size[1];
                    this.onResize?.(sz);
                    this.setDirtyCanvas(true, true);
                });
            };
		}

        // 2. LH_MultiTextSelector Logic
        if (nodeData.name === "LH_MultiTextSelector") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                setTimeout(() => {
                    // LH_MultiTextSelector uses 'widget_text'
                    setupSuperTextWidget(this, "widget_text", "batch_text", app);
                }, 1);
                
                return r;
            };
        }
        // 3. UniversalAIChat text behavior (user_material + instruction)
        if (nodeData.name === "UniversalAIChat") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                setTimeout(() => {
                    // user_material: when connected to active upstream -> read-only
                    // when disconnected or upstream muted/bypassed -> editable
                    setupSuperTextWidget(this, "user_material", "user_material", app);
                    // instruction: use the same logic for consistency
                    setupSuperTextWidget(this, "instruction", "instruction", app);
                }, 50);

                return r;
            };
        }
	},
});
