-- Copyright (c) 2026 Mohamed Alkobrosli
-- SPDX-License-Identifier: MIT
--
-- Pandoc filter for building the Word version of the paper from its reStructuredText source.
-- Pandoc's RST reader knows plain docutils but not Sphinx, so this filter supplies what
-- Sphinx does for the HTML and PDF builds:
--   * numbers tables and figures in document order and resolves :numref: with Sphinx's
--     default formats ("Table N", "Fig. N"), including a trailing panel letter;
--   * puts the same "Table N." / "Fig. N." label in front of each caption;
--   * maps the extension-less figure paths (figures/name.*) to the derived PNG files.
-- Equations are carried over as native Word equations but are not numbered (pandoc does not
-- number display math); the paper never refers to an equation by number.

local numbers = {}          -- identifier -> label text, e.g. "Table 2"
local counters = { table = 0, figure = 0 }

local function label_for(kind, n)
  if kind == "table" then return "Table " .. n end
  return "Fig. " .. n
end

-- Pass 1: number the floats in document order.
local function collect(doc)
  doc.blocks:walk({
    Div = function(div)
      if div.identifier ~= "" then
        local kind = nil
        for _, b in ipairs(div.content) do
          if b.t == "Table" then kind = "table"
          elseif b.t == "Figure" then kind = "figure" end
        end
        if kind then
          counters[kind] = counters[kind] + 1
          numbers[div.identifier] = label_for(kind, counters[kind])
        end
      end
    end,
  })
end

local function prefix_caption(caption, text)
  local inlines = pandoc.List({ pandoc.Strong({ pandoc.Str(text .. ".") }), pandoc.Space() })
  local blocks = caption.long
  if #blocks == 0 then
    caption.long = pandoc.List({ pandoc.Plain(inlines) })
  else
    local first = blocks[1]
    first.content = inlines .. first.content
  end
  return caption
end

function Pandoc(doc)
  collect(doc)
  local unresolved = {}
  local out = doc:walk({
    Div = function(div)
      local label = numbers[div.identifier]
      if label then
        for _, b in ipairs(div.content) do
          if b.t == "Table" or b.t == "Figure" then
            b.caption = prefix_caption(b.caption, label)
          end
        end
        return div
      end
    end,
    Image = function(img)
      img.src = img.src:gsub("%.%*$", ".png")
      return img
    end,
    Code = function(code)
      if code.attributes["role"] == "numref" then
        local label = numbers[code.text]
        if label == nil then
          unresolved[#unresolved + 1] = code.text
          return pandoc.Str("??" .. code.text .. "??")
        end
        return pandoc.Str(label)
      end
      if code.attributes["role"] == "ref" or code.attributes["role"] == "eq" then
        unresolved[#unresolved + 1] = code.text
      end
    end,
  })
  if #unresolved > 0 then
    io.stderr:write("ERROR: unresolved cross-references: " .. table.concat(unresolved, ", ") .. "\n")
    os.exit(1)
  end
  return out
end
